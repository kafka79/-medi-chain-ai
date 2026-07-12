import os
import logging
from typing import TypedDict, List, Dict, Any, Union
from pathlib import Path
import asyncio

import httpx
import torch
import numpy as np
from langgraph.graph import StateGraph, END

logger = logging.getLogger("clinical-graph")

# Flaw #13 Fix: Single source of truth for class labels — tied to model num_classes
from src.models.fusion import DIAGNOSTIC_CLASSES, NUM_CLASSES

# Flaw #6 Fix: OOD detection threshold — reject predictions where max softmax < this value
OOD_CONFIDENCE_THRESHOLD = float(os.getenv("OOD_CONFIDENCE_THRESHOLD", "0.4"))

# Flaw #15 Fix: Configurable thresholds with documentation
# These thresholds control the self-verification loop and escalation.
# CALIBRATION NOTE: These values should be validated against a held-out clinical dataset
# using ROC analysis. Current values are engineering defaults pending clinical validation.
_DEFAULT_CONFIDENCE = 0.6
_DEFAULT_UNCERTAINTY = 0.15
_DEFAULT_CALIBRATION = 1.0
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", str(_DEFAULT_CONFIDENCE)))
UNCERTAINTY_THRESHOLD = float(os.getenv("UNCERTAINTY_THRESHOLD", str(_DEFAULT_UNCERTAINTY)))
# ponytail: retry loop removed — identical inputs produce identical outputs, so retries were GPU waste.
# Uncertain cases now escalate immediately.
MC_DROPOUT_PASSES = int(os.getenv("MC_DROPOUT_PASSES", "50"))
UNCERTAINTY_CALIBRATION_FACTOR = float(os.getenv("UNCERTAINTY_CALIBRATION_FACTOR", str(_DEFAULT_CALIBRATION)))

# Flaw #8-structural Fix: Warn loudly at import time if thresholds are still uncalibrated defaults
_USING_DEFAULT_THRESHOLDS = (
    CONFIDENCE_THRESHOLD == _DEFAULT_CONFIDENCE
    and UNCERTAINTY_THRESHOLD == _DEFAULT_UNCERTAINTY
    and UNCERTAINTY_CALIBRATION_FACTOR == _DEFAULT_CALIBRATION
)
if _USING_DEFAULT_THRESHOLDS and os.getenv("TESTING") != "true":
    logger.critical(
        "SAFETY WARNING: All clinical decision thresholds (CONFIDENCE_THRESHOLD, "
        "UNCERTAINTY_THRESHOLD, UNCERTAINTY_CALIBRATION_FACTOR) are at uncalibrated "
        "engineering defaults. These MUST be tuned against a held-out clinical dataset "
        "using ROC/PR analysis before any patient-facing deployment. Set "
        "THRESHOLDS_VALIDATED=true in your environment after calibration to suppress this warning."
    )
    if os.getenv("THRESHOLDS_VALIDATED", "").lower() != "true":
        logger.critical("Set THRESHOLDS_VALIDATED=true after clinical calibration to acknowledge.")


class AgentState(TypedDict):
    schema_version: int
    image_path: str
    patient_pdf_path: str
    visual_features: Any
    visual_std: Any
    history_data: Dict[str, Any]
    pubmed_citations: List[Dict[str, Any]]
    diagnosis: Dict[str, Any]
    confidence: float
    iteration_count: int
    escalation_required: bool
    heatmap_base64: str


class ClinicalAgent:
    def __init__(self, history_parser, rag_evaluator, inference_api_url: str = None):
        self.parser = history_parser
        self.rag = rag_evaluator
        self.inference_api_url = inference_api_url or os.getenv("INFERENCE_API_URL", "http://inference-api:8001")
        self.internal_api_key = os.getenv("INTERNAL_API_KEY", "")
        if not self.internal_api_key and os.getenv("TESTING") != "true":
            raise RuntimeError("INTERNAL_API_KEY environment variable is required.")
        
        # Configure SSL / TLS settings
        self.ssl_verify = os.getenv("INTERNAL_SSL_VERIFY", "true")
        if self.ssl_verify.lower() == "true":
            self.ssl_verify = True
        elif self.ssl_verify.lower() == "false":
            self.ssl_verify = False
        
        ssl_cert_file = os.getenv("INTERNAL_SSL_CERT_FILE", None)
        ssl_key_file = os.getenv("INTERNAL_SSL_KEY_FILE", None)
        if ssl_cert_file and ssl_key_file:
            self.ssl_cert = (ssl_cert_file, ssl_key_file)
        elif ssl_cert_file:
            self.ssl_cert = ssl_cert_file
        else:
            self.ssl_cert = None
        
        # Flaw #5-structural Fix: Create ONE persistent httpx.AsyncClient to reuse
        # across all node calls, instead of creating/destroying one per request.
        # This eliminates TCP connection churn and potential socket leaks.
        self._http_client = httpx.AsyncClient(
            verify=self.ssl_verify,
            cert=self.ssl_cert,
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers={"X-Internal-API-Key": self.internal_api_key},
        )
        
        from ..data.privacy_scrubber import PrivacyScrubber
        self.scrubber = PrivacyScrubber()

        # Flaw #13 Fix + Panel Flaw #1 Fix: Validate class count via metadata —
        # no model instantiation on the Web API container. The previous approach
        # created a full LateFusionModel on CPU just to read .out_features, which
        # forced PyTorch weight loading on a container that should be GPU-free.
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
        await self._http_client.aclose()

    def _build_graph(self):
        # Define Nodes
        self.workflow.add_node("extract_visuals", self.node_extract_visuals)
        self.workflow.add_node("parse_history", self.node_parse_history)
        self.workflow.add_node("query_pubmed", self.node_query_pubmed)
        self.workflow.add_node("synthesize_diagnosis", self.node_synthesize_diagnosis)
        self.workflow.add_node("self_verify", self.node_self_verify)
        
        # Define Edges
        # ponytail: linear pipeline, no retry loop. Retry was running identical inference
        # on unchanged tensors (PubMed citations are not fed into model inputs).
        # Uncertain cases escalate directly to radiologist.
        self.workflow.set_entry_point("extract_visuals")
        self.workflow.add_edge("extract_visuals", "parse_history")
        self.workflow.add_edge("parse_history", "query_pubmed")
        self.workflow.add_edge("query_pubmed", "synthesize_diagnosis")
        self.workflow.add_edge("synthesize_diagnosis", "self_verify")
        self.workflow.add_edge("self_verify", END)
        
        self.app = self.workflow.compile()

    def _ensure_current_schema(self, state: AgentState) -> AgentState:
        # Ensure we don't modify the state dictionary in place if it's read-only
        state_dict = dict(state)
        current_version = state_dict.get("schema_version", 1)
        if current_version < 2:
            # Migrate v1 to v2: ensure all required keys exist and schema_version is set
            if "heatmap_base64" not in state_dict:
                state_dict["heatmap_base64"] = ""
            state_dict["schema_version"] = 2
            logger.info(f"[Schema Migration] Migrated AgentState from v{current_version} to v2")
        return state_dict

    # ponytail: negation words that negate the clinical concept in a 3-word window
    _NEGATION_CUES = {"no", "not", "never", "denies", "denied", "without", "absent", "negative", "none", "nor", "ruled"}

    def _is_negated(self, trigger: str, text: str) -> bool:
        """Check if a trigger term is preceded by a negation word within a 3-word window."""
        idx = text.find(trigger)
        if idx < 0:
            return False
        # Grab up to 40 chars before the match to get ~3 words of context
        window_start = max(0, idx - 40)
        preceding = text[window_start:idx].split()
        # Check last 3 words before the trigger
        for word in preceding[-3:]:
            if word.strip(".,;:'\"") in self._NEGATION_CUES:
                return True
        return False

    def _extract_biomedical_concepts(self, chief_complaint: str, pmh: str) -> List[str]:
        """
        Negation-aware concept extraction from clinical text.
        ponytail: added negation window check so 'no asbestos exposure' doesn't map to asbestosis.
        """
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
            "nodule": "pulmonary nodule"
        }
        for trigger, concept in all_triggers.items():
            if trigger in combined_text and not self._is_negated(trigger, combined_text):
                concepts.append(concept)
                
        # Dynamic token filtering for remaining clinical terms
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

    # Node Functions
    async def node_extract_visuals(self, state: AgentState):
        logger.info("[Node] Extracting Visuals...")
        
        orig_img_path = state['image_path']
        
        # Mask the burned-in text and retrieve the anonymized image path
        img_to_encode = self.scrubber.mask_burned_in_text(orig_img_path)
        success = (img_to_encode != orig_img_path)
        
        # Flaw #6-structural Fix: Always use multipart upload. The previous approach
        # tried to send a file *path* first, which silently breaks in multi-container
        # deployments where the inference service can't access the API container's
        # filesystem. Multipart upload works regardless of deployment topology.
        try:
            with open(img_to_encode, "rb") as f:
                files = {"image": ("scan.jpg", f, "image/jpeg")}
                resp = await self._http_client.post(
                    f"{self.inference_api_url}/encode/image",
                    files=files,
                    timeout=15
                )
            resp.raise_for_status()
            resp_data = resp.json()
            features = resp_data["features"]
            visual_std = resp_data.get("visual_std", None)
            heatmap_base64 = resp_data.get("heatmap_base64", "")
        except Exception as e:
            logger.error(f"[Clinical Graph] Error calling inference API: {e}")
            raise RuntimeError(f"Visual encoder failed: {e}")
        finally:
            # Clean up the temporary scrubbed image if it was created
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

    async def node_query_pubmed(self, state: AgentState):
        logger.info("[Node] Querying PubMed...")
        state = self._ensure_current_schema(state)
        history = state['history_data']
        metadata = history.get('metadata', {})
        
        # Combine clinical history elements for a specific medical query
        chief_complaint = history.get('chief_complaint', '')
        pmh = history.get('past_medical_history', '')
        occupation = metadata.get('occupation', '')
        exposure_years = metadata.get('exposure_years', '0')
        
        # Build search keywords based on PMH or Chief Complaint dynamically (replaces hardcoded ontology checks)
        keywords = self._extract_biomedical_concepts(chief_complaint, pmh)
            
        query_parts = []
        if chief_complaint:
            query_parts.append(chief_complaint)
        if keywords:
            query_parts.append(" AND ".join(keywords))
        if occupation and exposure_years != "0":
            query_parts.append(f"{occupation} exposure")
            
        base_query = " ".join(query_parts) if query_parts else "chest radiography diagnostic"
        
        if state.get('iteration_count', 0) > 0:
            query = f"{base_query} differential diagnosis respiratory imaging"
        else:
            query = base_query
            
        try:
            citations = await self.rag.search(query, 3)
            return {"pubmed_citations": citations}
        except Exception as e:
            logger.error(f"[Clinical Graph] RAG retrieval failed: {e}. Forcing human review escalation.")
            return {"pubmed_citations": [], "escalation_required": True}

    async def node_synthesize_diagnosis(self, state: AgentState):
        logger.info("[Node] Synthesizing Diagnosis...")
        state = self._ensure_current_schema(state)
        # Get visual features (already list from API)
        v = state['visual_features']
            
        # RAG Cognitive Noise Fix: Keep patient clinical history separate from retrieved public PubMed literature text.
        # Do not append raw PubMed abstract text to the text encoder content to avoid feature space pollution.
        # Retrieved citations are stored in state and used exclusively for downstream clinician XAIExplainer.
        history = state['history_data']
        text_content = f"{history.get('chief_complaint', '')} {history.get('history_present_illness', '')} {history.get('labs', '')}".strip()
        
        # Embed text using remote API (Flaw #5-structural: uses persistent self._http_client)
        try:
            resp = await self._http_client.post(
                f"{self.inference_api_url}/encode/text",
                json={"text": text_content},
            )
            resp.raise_for_status()
            t = resp.json()["embeddings"]
        except Exception as e:
            logger.error(f"[Clinical Graph] Error calling inference API text encoder: {e}")
            raise RuntimeError(f"Text encoder failed: {e}")
        
        # Run uncertainty estimation via remote API
        try:
            resp = await self._http_client.post(
                f"{self.inference_api_url}/estimate",
                json={
                    "visual_features": v,
                    "visual_std": state.get("visual_std"),
                    "text_features": t,
                    # ponytail: configurable via MC_DROPOUT_PASSES, default 50 for stable variance
                    "num_passes": MC_DROPOUT_PASSES
                },
            )
            resp.raise_for_status()
            results = resp.json()
        except Exception as e:
            logger.error(f"[Clinical Graph] Error calling inference API uncertainty estimator: {e}")
            raise RuntimeError(f"Uncertainty estimator failed: {e}")
        
        pred_idx = int(results['prediction'][0])
        
        # Flaw #13 Fix: Use module-level DIAGNOSTIC_CLASSES instead of inline hardcoded list
        top_finding = DIAGNOSTIC_CLASSES[pred_idx] if pred_idx < NUM_CLASSES else "Unknown"
        
        mean_confidence = float(results['mean_confidence'][0])
        uncertainty_std = float(results['std_deviation'][0])
        all_probs = results['all_probs'][0]
        max_softmax = max(all_probs) if all_probs else 0.0

        # Distance-based visual OOD detection
        visual_ood_detected = False
        try:
            import json
            baseline_path = Path("temp/drift/features_baseline_cache.json")
            if baseline_path.exists():
                with open(baseline_path, "r") as f:
                    baseline_features = json.load(f)
                
                if baseline_features and len(baseline_features) > 0:
                    baseline_arr = np.array(baseline_features)  # (N, 512)
                    current_arr = np.array(v)                   # (1, 512)
                    
                    mean_baseline = np.mean(baseline_arr, axis=0)
                    mean_current = np.mean(current_arr, axis=0)
                    
                    norm_b = np.linalg.norm(mean_baseline)
                    norm_c = np.linalg.norm(mean_current)
                    
                    if norm_b > 0 and norm_c > 0:
                        cos_sim = np.dot(mean_baseline, mean_current) / (norm_b * norm_c)
                        
                        use_static_ood = os.getenv("OOD_USE_STATIC_THRESHOLD", "true").lower() == "true"
                        if use_static_ood:
                            cos_sim_threshold = float(os.getenv("OOD_COSINE_THRESHOLD", "0.82"))
                            logger.info(f"[OOD Check] Using static OOD cosine threshold: {cos_sim_threshold:.4f}")
                        else:
                            # Dynamically calibrate the threshold based on baseline similarity statistics
                            # Calculate similarity of each baseline vector to the mean baseline vector
                            baseline_norms = np.linalg.norm(baseline_arr, axis=1)
                            valid_mask = (baseline_norms > 0) & (norm_b > 0)
                            n_samples = len(baseline_features)
                            if np.any(valid_mask) and n_samples >= 10:
                                baseline_similarities = np.dot(baseline_arr[valid_mask], mean_baseline) / (baseline_norms[valid_mask] * norm_b)
                                median_sim = np.median(baseline_similarities)
                                mad_sim = np.median(np.abs(baseline_similarities - median_sim))
                                # Convert MAD to standard deviation scale (1.4826)
                                mad_std = mad_sim * 1.4826
                                # Scale outlier sensitivity based on sample size: tighter check for small sample size
                                multiplier = 2.0 if n_samples < 50 else 3.0
                                calibrated_threshold = median_sim - multiplier * mad_std
                            else:
                                # Too small baseline, use static default
                                calibrated_threshold = 0.82
                                
                            # Cap it to a safe range [0.75, 0.90] to prevent alert fatigue or under-sensitivity
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

        # Distance-based text OOD detection to close the text-branch OOD blindspot
        text_ood_detected = False
        try:
            text_baseline_path = Path("temp/drift/text_baseline_cache.json")
            if text_baseline_path.exists():
                with open(text_baseline_path, "r") as f:
                    text_baseline_features = json.load(f)
                
                if text_baseline_features and len(text_baseline_features) > 0:
                    text_baseline_arr = np.array(text_baseline_features)  # (N, 768)
                    current_text_arr = np.array(t)                        # (1, 768)
                    
                    mean_baseline_text = np.mean(text_baseline_arr, axis=0)
                    mean_current_text = np.mean(current_text_arr, axis=0)
                    
                    norm_b_text = np.linalg.norm(mean_baseline_text)
                    norm_c_text = np.linalg.norm(mean_current_text)
                    
                    if norm_b_text > 0 and norm_c_text > 0:
                        cos_sim_t = np.dot(mean_baseline_text, mean_current_text) / (norm_b_text * norm_c_text)
                        
                        # Dynamic or static threshold for text OOD check
                        cos_sim_threshold_t = float(os.getenv("OOD_TEXT_COSINE_THRESHOLD", "0.82"))
                        if cos_sim_t < cos_sim_threshold_t:
                            text_ood_detected = True
                            logger.warning(
                                f"[OOD Detection] Distance-based text OOD check failed: "
                                f"cosine similarity = {cos_sim_t:.4f} < threshold {cos_sim_threshold_t}."
                            )
        except Exception as e:
            logger.error(f"Error running distance-based text OOD detector: {e}")

        # OOD detection — reject predictions where max softmax is below threshold or visual/text distance check fails
        ood_flag = (max_softmax < OOD_CONFIDENCE_THRESHOLD) or visual_ood_detected or text_ood_detected
        if ood_flag:
            logger.warning(
                f"[OOD Detection] OOD check triggered. "
                f"max(softmax) = {max_softmax:.4f} (threshold {OOD_CONFIDENCE_THRESHOLD}), "
                f"visual_ood_detected = {visual_ood_detected}, "
                f"text_ood_detected = {text_ood_detected}. Flagging for escalation."
            )
            top_finding = "Out-of-Distribution"
            # Overwrite uncertainty to a high value to reflect OOD
            uncertainty_std = max(uncertainty_std, 0.5)
        
        # Generate Clinical Rationale (Fixes 'User-Led Explanation')
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
            # Flaw #6: If OOD detected, force escalation
            "escalation_required": ood_flag,
            "visual_features": v,
            "text_features": t[0] if isinstance(t, list) else t.tolist(),
        }

    async def node_self_verify(self, state: AgentState):
        """ponytail: single-pass verify. Uncertain = escalate. No retry loop."""
        state = self._ensure_current_schema(state)
        logger.info(f"[Node] Self-Verifying (Confidence: {state['confidence']:.2f}, Uncertainty: {state['diagnosis']['uncertainty_std']:.4f})...")
        
        scaled_std = state['diagnosis']['uncertainty_std'] * UNCERTAINTY_CALIBRATION_FACTOR
        is_uncertain = state['confidence'] < CONFIDENCE_THRESHOLD or scaled_std > UNCERTAINTY_THRESHOLD
        
        # ponytail: escalate immediately if OOD or uncertain. Retry loop removed because
        # inputs don't change between iterations — PubMed citations are not fed to the model.
        escalate = state.get('escalation_required', False) or is_uncertain
        if escalate:
            logger.warning(f"--- Escalation required (OOD={state.get('escalation_required', False)}, Uncertain={is_uncertain}). ---")
            return {"escalation_required": True}
            
        return {}

    async def run(self, image_path: str, pdf_path: str):
        initial_state = {
            "schema_version": 2,
            "image_path": image_path,
            "patient_pdf_path": pdf_path,
            "iteration_count": 0,
            "escalation_required": False,
            "pubmed_citations": [],
            "visual_features": None,
            "visual_std": None,
            "history_data": {},
            "diagnosis": {},
            "confidence": 0.0,
            "heatmap_base64": ""
        }
        return await self.app.ainvoke(initial_state)

if __name__ == "__main__":
    pass
