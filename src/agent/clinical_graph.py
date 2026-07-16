import os
import logging
from typing import TypedDict, List, Dict, Any, Union, Optional
from pathlib import Path
import asyncio
import contextvars

import httpx
import torch
import numpy as np
from langgraph.graph import StateGraph, END
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)

from src.config.settings import get_clinical_thresholds, get_inference_config
from src.models.fusion import DIAGNOSTIC_CLASSES, NUM_CLASSES

logger = logging.getLogger("clinical-graph")

# Load validated clinical thresholds
clinical_config = get_clinical_thresholds()
inference_config = get_inference_config()

CONFIDENCE_THRESHOLD = clinical_config.confidence_threshold
UNCERTAINTY_THRESHOLD = clinical_config.uncertainty_threshold
UNCERTAINTY_CALIBRATION_FACTOR = clinical_config.uncertainty_calibration_factor
OOD_CONFIDENCE_THRESHOLD = clinical_config.ood_confidence_threshold
OOD_USE_STATIC_THRESHOLD = clinical_config.ood_use_static_threshold
OOD_COSINE_THRESHOLD = clinical_config.ood_cosine_threshold
OOD_TEXT_COSINE_THRESHOLD = clinical_config.ood_text_cosine_threshold
MC_DROPOUT_PASSES = clinical_config.mc_dropout_passes

# Warn if using uncalibrated defaults
_USING_DEFAULT_THRESHOLDS = not clinical_config.thresholds_validated
if _USING_DEFAULT_THRESHOLDS and os.getenv("TESTING") != "true":
    logger.critical(
        "SAFETY WARNING: Clinical decision thresholds are at uncalibrated "
        "engineering defaults. These MUST be tuned against a held-out clinical dataset "
        "using ROC/PR analysis before any patient-facing deployment. Set "
        "CLINICAL_THRESHOLDS_VALIDATED=true in your environment after calibration."
    )
    if os.getenv("CLINICAL_THRESHOLDS_VALIDATED", "").lower() != "true":
        logger.critical("Set CLINICAL_THRESHOLDS_VALIDATED=true after clinical calibration to acknowledge.")


class ClinicalInferenceError(Exception):
    """Raised when the clinical inference API fails, requiring safe escalation."""
    pass


class AgentState(TypedDict):
    schema_version: int
    image_path: str
    patient_pdf_path: str
    visual_features: Any
    visual_std: Any
    text_features: Any
    history_data: Dict[str, Any]
    pubmed_citations: List[Dict[str, Any]]
    diagnosis: Dict[str, Any]
    confidence: float
    iteration_count: int
    escalation_required: bool
    inference_failed: bool
    heatmap_base64: str
    idempotency_key: Optional[str]


class CircuitBreakerState:
    def __init__(self):
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open
    
    def record_success(self):
        self.success_count += 1
        if self.state == "half-open" and self.success_count >= 3:
            self.state = "closed"
            self.failure_count = 0
            logger.info("Circuit breaker closed after successful recovery")
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= 5:
            self.state = "open"
            logger.warning("Circuit breaker opened due to repeated failures")


class ResilientHTTPClient:
    """HTTP client with circuit breaker, request hedging, and timeout budget."""
    
    def __init__(self, config: Any, base_url: str, api_key: str, ssl_verify, ssl_cert):
        self.config = config
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        
        self._client = httpx.AsyncClient(
            verify=ssl_verify,
            cert=ssl_cert,
            timeout=httpx.Timeout(config.request_timeout_seconds, connect=config.connect_timeout_seconds),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers={"X-Internal-API-Key": api_key},
        )
    
    def _get_circuit_breaker(self, endpoint: str) -> CircuitBreakerState:
        if endpoint not in self.circuit_breakers:
            self.circuit_breakers[endpoint] = CircuitBreakerState()
        return self.circuit_breakers[endpoint]
    
    def _check_circuit_breaker(self, endpoint: str) -> bool:
        cb = self._get_circuit_breaker(endpoint)
        if cb.state == "open":
            if time.time() - cb.last_failure_time > self.config.circuit_breaker_open_seconds:
                cb.state = "half-open"
                cb.success_count = 0
                logger.info(f"Circuit breaker for {endpoint} entering half-open state")
                return True
            return False
        return True
    
    async def _request_with_hedging(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Make request with hedging: if first request takes > hedging_delay, fire second."""
        if not self._check_circuit_breaker(endpoint):
            raise httpx.ConnectError(f"Circuit breaker open for {endpoint}")
        
        hedging_delay = self.config.request_timeout_seconds / self.config.hedging_delay_factor
        first_response: Optional[httpx.Response] = None
        first_error: Optional[Exception] = None
        
        async def make_request():
            return await self._client.request(method, f"{self.base_url}{endpoint}", **kwargs)
        
        # Start first request
        task1 = asyncio.create_task(make_request())
        
        try:
            # Wait for hedging delay or completion
            done, pending = await asyncio.wait(
                [task1], 
                timeout=hedging_delay,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            if task1 in done:
                first_response = task1.result()
                if first_response.is_success:
                    self._get_circuit_breaker(endpoint).record_success()
                    return first_response
                first_error = Exception(f"HTTP {first_response.status_code}")
            else:
                # Fire hedging request
                task2 = asyncio.create_task(make_request())
                done, _ = await asyncio.wait([task1, task2], return_when=asyncio.FIRST_COMPLETED)
                
                for task in done:
                    try:
                        resp = task.result()
                        if resp.is_success:
                            self._get_circuit_breaker(endpoint).record_success()
                            # Cancel the other
                            for t in [task1, task2]:
                                if not t.done():
                                    t.cancel()
                            return resp
                        first_error = Exception(f"HTTP {resp.status_code}")
                    except Exception as e:
                        first_error = e
                
                # Both failed
                self._get_circuit_breaker(endpoint).record_failure()
                raise first_error or Exception("Both hedged requests failed")
        
        except asyncio.CancelledError:
            if not task1.done():
                task1.cancel()
            raise
        except Exception:
            self._get_circuit_breaker(endpoint).record_failure()
            raise
    
    async def post(self, endpoint: str, **kwargs) -> httpx.Response:
        return await self._request_with_hedging("POST", endpoint, **kwargs)
    
    async def get(self, endpoint: str, **kwargs) -> httpx.Response:
        return await self._request_with_hedging("GET", endpoint, **kwargs)
    
    async def close(self):
        await self._client.aclose()


import time


class ClinicalAgent:
    def __init__(self, history_parser, rag_evaluator, inference_api_url: str = None):
        self.parser = history_parser
        self.rag = rag_evaluator
        self.inference_api_url = inference_api_url or os.getenv("INFERENCE_API_URL", "http://inference-api:8001")
        self.internal_api_key = os.getenv("INTERNAL_API_KEY", "")
        if not self.internal_api_key and os.getenv("TESTING") != "true":
            raise RuntimeError("INTERNAL_API_KEY environment variable is required.")
        
        # Configure SSL / TLS settings
        ssl_verify = os.getenv("INTERNAL_SSL_VERIFY", "true")
        if ssl_verify.lower() == "true":
            ssl_verify = True
        elif ssl_verify.lower() == "false":
            ssl_verify = False
        
        ssl_cert_file = os.getenv("INTERNAL_SSL_CERT_FILE", None)
        ssl_key_file = os.getenv("INTERNAL_SSL_KEY_FILE", None)
        if ssl_cert_file and ssl_key_file:
            ssl_cert = (ssl_cert_file, ssl_key_file)
        elif ssl_cert_file:
            ssl_cert = ssl_cert_file
        else:
            ssl_cert = None
        
        # Initialize resilient HTTP client
        self._http_client = ResilientHTTPClient(
            inference_config, self.inference_api_url, self.internal_api_key, ssl_verify, ssl_cert
        )
        
        from ..data.privacy_scrubber import PrivacyScrubber
        self.scrubber = PrivacyScrubber()

        # Validate class count via metadata
        from src.models.fusion import get_model_num_classes
        _expected_classes = get_model_num_classes()
        assert _expected_classes == NUM_CLASSES, (
            f"FATAL: Model configured for {_expected_classes} output classes but "
            f"DIAGNOSTIC_CLASSES has {NUM_CLASSES}. These MUST match."
        )
        
        self.workflow = StateGraph(AgentState)
        self._build_graph()

    async def close(self):
        """Gracefully close the persistent HTTP client. Call on shutdown."""
        await self._http_client.close()

    def _build_graph(self):
        self.workflow.add_node("extract_visuals", self.node_extract_visuals)
        self.workflow.add_node("parse_history", self.node_parse_history)
        self.workflow.add_node("synthesize_diagnosis", self.node_synthesize_diagnosis)
        self.workflow.add_node("self_verify", self.node_self_verify)
        
        self.workflow.set_entry_point("extract_visuals")
        self.workflow.add_edge("extract_visuals", "parse_history")
        self.workflow.add_edge("parse_history", "synthesize_diagnosis")
        self.workflow.add_edge("synthesize_diagnosis", "self_verify")
        self.workflow.add_edge("self_verify", END)
        
        self.app = self.workflow.compile()

    def _ensure_current_schema(self, state: AgentState) -> AgentState:
        state_dict = dict(state)
        current_version = state_dict.get("schema_version", 1)
        if current_version < 2:
            if "heatmap_base64" not in state_dict:
                state_dict["heatmap_base64"] = ""
            if "idempotency_key" not in state_dict:
                state_dict["idempotency_key"] = None
            state_dict["schema_version"] = 2
            logger.info(f"[Schema Migration] Migrated AgentState from v{current_version} to v2")
        return state_dict

    _NEGATION_CUES = {"no", "not", "never", "denies", "denied", "without", "absent", "negative", "none", "nor", "ruled"}

    def _is_negated(self, trigger: str, text: str) -> bool:
        idx = text.find(trigger)
        if idx < 0:
            return False
        window_start = max(0, idx - 40)
        preceding = text[window_start:idx].split()
        for word in preceding[-3:]:
            if word.strip(".,;:'\"") in self._NEGATION_CUES:
                return True
        return False

    def _extract_biomedical_concepts(self, chief_complaint: str, pmh: str) -> List[str]:
        import re
        combined_text = f"{chief_complaint} {pmh}".lower()
        concepts = []
        
        all_triggers = {
            "asbest": "asbestosis", "silic": "silicosis",
            "coal": "coal workers' pneumoconiosis", "dust": "dust exposure lung disease",
            "quarry": "silicosis", "mining": "pneumoconiosis",
            "berylli": "berylliosis", "cotton": "byssinosis",
            "iron": "siderosis", "sandblast": "silicosis",
            "pneumonia": "pneumonia", "tuberculosis": "tuberculosis",
            "tb": "tuberculosis", "bronchitis": "bronchitis",
            "sarcoid": "sarcoidosis", "effusion": "pleural effusion",
            "cancer": "lung malignancy", "tumor": "lung neoplasm",
            "nodule": "pulmonary nodule", "mass": "lung mass",
            "adenocarcinoma": "lung adenocarcinoma", "squamous": "squamous cell carcinoma",
            "copd": "chronic obstructive pulmonary disease", "emphysema": "emphysema",
            "fibrosis": "pulmonary fibrosis", "ild": "interstitial lung disease",
            "pulmonary embolism": "pulmonary embolism", "pe": "pulmonary embolism",
        }
        for trigger, concept in all_triggers.items():
            if trigger in combined_text and not self._is_negated(trigger, combined_text):
                concepts.append(concept)
                
        stop_words = {
            "and", "the", "with", "for", "from", "on", "in", "to", "of", "a", "an", "is", "was",
            "history", "patient", "clinical", "presentation", "years", "old", "male", "female",
            "complains", "presenting", "reported", "shows", "findings", "reveals", "normal", "abnormal"
        }
        clean_text = re.sub(r"[^a-zA-Z\s]", " ", combined_text)
        words = clean_text.split()
        candidate_terms = [w for w in words if len(w) > 4 and w not in stop_words]
        
        for term in candidate_terms:
            if len(concepts) >= 6:
                break
            if not any(term in c or c in term for c in concepts):
                if not self._is_negated(term, combined_text):
                    concepts.append(term)
        
        return list(set(concepts))

    async def node_extract_visuals(self, state: AgentState):
        logger.info("[Node] Extracting Visuals...")
        
        orig_img_path = state['image_path']
        img_to_encode = self.scrubber.mask_burned_in_text(orig_img_path)
        success = (img_to_encode != orig_img_path)
        
        try:
            with open(img_to_encode, "rb") as f:
                files = {"image": ("scan.jpg", f, "image/jpeg")}
                resp = await self._http_client.post(
                    "/encode/image",
                    files=files,
                )
            resp.raise_for_status()
            resp_data = resp.json()
            features = resp_data["features"]
            visual_std = resp_data.get("visual_std", None)
            heatmap_base64 = resp_data.get("heatmap_base64", "")
        except Exception as e:
            logger.error(f"[Clinical Graph] Error calling inference API: {e}. Raising ClinicalInferenceError for upstream circuit breaking.")
            raise ClinicalInferenceError(f"Inference API failed during extract_visuals: {e}") from e
        finally:
            if success and os.path.exists(img_to_encode) and img_to_encode != orig_img_path:
                try:
                    os.remove(img_to_encode)
                except Exception as e:
                    logger.warning(f"[Clinical Graph] Warning: failed to clean up temp scrubbed image: {e}")
                
        return {"visual_features": features, "visual_std": visual_std, "heatmap_base64": heatmap_base64}

    async def node_parse_history(self, state: AgentState):
        logger.info("[Node] Parsing Patient History...")
        state = self._ensure_current_schema(state)
        path = state['patient_pdf_path']
        loop = asyncio.get_running_loop()
        if path.lower().endswith(".json"):
            import json
            def load_json():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            history = await loop.run_in_executor(None, load_json)
        else:
            raw_history = await loop.run_in_executor(None, self.parser.parse_pdf, path)
            history = await loop.run_in_executor(None, self.scrubber.scrub_history_data, raw_history)
        
        return {"history_data": history}



    async def node_synthesize_diagnosis(self, state: AgentState):
        logger.info("[Node] Synthesizing Diagnosis...")
        state = self._ensure_current_schema(state)
        v = state['visual_features']
            
        # RAG Cognitive Noise Fix: Keep patient clinical history separate from retrieved public PubMed literature text.
        history = state['history_data']
        text_content = f"{history.get('chief_complaint', '')} {history.get('history_present_illness', '')} {history.get('labs', '')}".strip()
        
        if state.get("inference_failed", False):
            return {
                "diagnosis": {
                    "top_finding": "Unknown - Inference API Failure",
                    "probabilities": {},
                },
                "confidence": 0.0,
                "escalation_required": True
            }

        # Embed text using remote API
        try:
            resp = await self._http_client.post(
                "/encode/text",
                json={"text": text_content},
            )
            resp.raise_for_status()
            t = resp.json()["embeddings"]
        except Exception as e:
            logger.error(f"[Clinical Graph] Error calling inference API text encoder: {e}. Degrading gracefully.")
            return {
                "diagnosis": {
                    "top_finding": "Unknown - Inference API Failure (Text)",
                    "probabilities": {},
                },
                "confidence": 0.0,
                "escalation_required": True,
                "inference_failed": True
            }
        
        # Run uncertainty estimation via remote API
        try:
            resp = await self._http_client.post(
                "/estimate",
                json={
                    "visual_features": v,
                    "visual_std": state.get("visual_std"),
                    "text_features": t,
                    "num_passes": MC_DROPOUT_PASSES
                },
            )
            resp.raise_for_status()
            results = resp.json()
        except Exception as e:
            logger.error(f"[Clinical Graph] Error calling inference API uncertainty estimator: {e}. Raising ClinicalInferenceError.")
            raise ClinicalInferenceError(f"Inference API failed during estimate: {e}") from e
        
        pred_idx = int(results['prediction'][0])
        
        top_finding = DIAGNOSTIC_CLASSES[pred_idx] if pred_idx < NUM_CLASSES else "Unknown"
        
        mean_confidence = float(results['mean_confidence'][0])
        uncertainty_std = float(results['std_deviation'][0])
        all_probs = results['all_probs'][0]
        max_softmax = max(all_probs) if all_probs else 0.0

        # Distance-based visual OOD detection
        visual_ood_detected = False
        try:
            import json
            centroid_path = Path("temp/drift/features_centroid.json")
            if centroid_path.exists():
                with open(centroid_path, "r") as f:
                    centroid_data = json.load(f)
                
                if centroid_data and "centroid" in centroid_data:
                    mean_baseline = np.array(centroid_data["centroid"])
                    current_arr = np.array(v)
                    mean_current = np.mean(current_arr, axis=0)
                    
                    norm_b = np.linalg.norm(mean_baseline)
                    norm_c = np.linalg.norm(mean_current)
                    
                    if norm_b > 0 and norm_c > 0:
                        cos_sim = np.dot(mean_baseline, mean_current) / (norm_b * norm_c)
                        
                        if OOD_USE_STATIC_THRESHOLD:
                            cos_sim_threshold = OOD_COSINE_THRESHOLD
                            logger.info(f"[OOD Check] Using static OOD cosine threshold: {cos_sim_threshold:.4f}")
                        else:
                            median_sim = centroid_data.get("median_sim", 0.0)
                            mad_std = centroid_data.get("mad_std", 0.0)
                            n_samples = centroid_data.get("count", 0)
                            if mad_std > 0 and n_samples >= 10:
                                multiplier = 2.0 if n_samples < 50 else 3.0
                                calibrated_threshold = median_sim - multiplier * mad_std
                            else:
                                calibrated_threshold = 0.82
                                
                            cos_sim_threshold = float(os.getenv("OOD_COSINE_THRESHOLD", str(max(0.75, min(0.90, calibrated_threshold)))))
                            logger.info(f"[OOD Calibration] Dynamically calibrated OOD threshold to {cos_sim_threshold:.4f} (calibrated={calibrated_threshold:.4f}, samples={n_samples}).")
                            
                        if cos_sim < cos_sim_threshold:
                            visual_ood_detected = True
                            logger.warning(
                                f"[OOD Detection] Distance-based visual OOD check failed: "
                                f"cosine similarity = {cos_sim:.4f} < threshold {cos_sim_threshold}."
                            )
        except Exception as e:
            logger.error(f"Error running distance-based OOD detector: {e}")

        # Distance-based text OOD detection
        text_ood_detected = False
        try:
            text_centroid_path = Path("temp/drift/text_centroid.json")
            if text_centroid_path.exists():
                with open(text_centroid_path, "r") as f:
                    centroid_data_t = json.load(f)
                
                if centroid_data_t and "centroid" in centroid_data_t:
                    mean_baseline_text = np.array(centroid_data_t["centroid"])
                    current_text_arr = np.array(t)
                    mean_current_text = np.mean(current_text_arr, axis=0)
                    
                    norm_b_text = np.linalg.norm(mean_baseline_text)
                    norm_c_text = np.linalg.norm(mean_current_text)
                    
                    if norm_b_text > 0 and norm_c_text > 0:
                        cos_sim_t = np.dot(mean_baseline_text, mean_current_text) / (norm_b_text * norm_c_text)
                        
                        if cos_sim_t < OOD_TEXT_COSINE_THRESHOLD:
                            text_ood_detected = True
                            logger.warning(
                                f"[OOD Detection] Distance-based text OOD check failed: "
                                f"cosine similarity = {cos_sim_t:.4f} < threshold {OOD_TEXT_COSINE_THRESHOLD}."
                            )
        except Exception as e:
            logger.error(f"Error running distance-based text OOD detector: {e}")

        # OOD detection
        ood_flag = (max_softmax < OOD_CONFIDENCE_THRESHOLD) or visual_ood_detected or text_ood_detected
        if ood_flag:
            logger.warning(
                f"[OOD Detection] OOD check triggered. "
                f"max(softmax) = {max_softmax:.4f} (threshold {OOD_CONFIDENCE_THRESHOLD}), "
                f"visual_ood_detected = {visual_ood_detected}, "
                f"text_ood_detected = {text_ood_detected}. Flagging for escalation."
            )
            top_finding = "Out-of-Distribution"
            uncertainty_std = max(uncertainty_std, 0.5)
        
        # Generate Clinical Rationale
        from src.monitoring.xai_explainer import XAIExplainer
        explainer = XAIExplainer()
        
        loop = asyncio.get_running_loop()
        rationale = await loop.run_in_executor(
            None, 
            explainer.explain,
            top_finding, 
            mean_confidence,
            uncertainty_std,
            all_probs,
            history,
            state.get('pubmed_citations', [])
        )
        
        diagnosis = {
            "top_finding": top_finding,
            "rationale": rationale,
            "probabilities": all_probs,
            "uncertainty_std": uncertainty_std,
            "ood_detected": ood_flag,
        }
        
        return {
            "diagnosis": diagnosis, 
            "confidence": mean_confidence,
            "escalation_required": ood_flag,
            "visual_features": v,
            "text_features": t[0] if isinstance(t, list) else t.tolist(),
        }

    async def node_self_verify(self, state: AgentState):
        """Single-pass verify. Uncertain = escalate. No retry loop."""
        state = self._ensure_current_schema(state)
        logger.info(f"[Node] Self-Verifying (Confidence: {state['confidence']:.2f}, Uncertainty: {state['diagnosis']['uncertainty_std']:.4f})...")
        
        scaled_std = state['diagnosis']['uncertainty_std'] * UNCERTAINTY_CALIBRATION_FACTOR
        is_uncertain = state['confidence'] < CONFIDENCE_THRESHOLD or scaled_std > UNCERTAINTY_THRESHOLD
        
        escalate = state.get('escalation_required', False) or is_uncertain
        if escalate:
            logger.warning(f"--- Escalation required (OOD={state.get('escalation_required', False)}, Uncertain={is_uncertain}). ---")
            return {"escalation_required": True}
            
        return {}

    async def run(self, image_path: str, pdf_path: str, idempotency_key: Optional[str] = None):
        initial_state = {
            "schema_version": 2,
            "image_path": image_path,
            "patient_pdf_path": pdf_path,
            "iteration_count": 0,
            "escalation_required": False,
            "pubmed_citations": [],
            "visual_features": None,
            "visual_std": None,
            "text_features": None,
            "history_data": {},
            "diagnosis": {},
            "confidence": 0.0,
            "heatmap_base64": "",
            "idempotency_key": idempotency_key,
        }
        return await self.app.ainvoke(initial_state)

if __name__ == "__main__":
    pass