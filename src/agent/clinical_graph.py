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
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
UNCERTAINTY_THRESHOLD = float(os.getenv("UNCERTAINTY_THRESHOLD", "0.15"))
MAX_RETRY_ITERATIONS = int(os.getenv("MAX_RETRY_ITERATIONS", "3"))
UNCERTAINTY_CALIBRATION_FACTOR = float(os.getenv("UNCERTAINTY_CALIBRATION_FACTOR", "1.0"))


class AgentState(TypedDict):
    image_path: str
    patient_pdf_path: str
    visual_features: Any
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
        self.ssl_verify = os.getenv("INTERNAL_SSL_VERIFY", "false")
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
        
        from ..data.privacy_scrubber import PrivacyScrubber
        self.scrubber = PrivacyScrubber()

        # Flaw #13 Fix: Assert class count matches the fusion model at construction time
        from src.models.fusion import LateFusionModel
        _tmp_model = LateFusionModel()
        # Extract num_classes from the classifier's final Linear layer
        classifier_out_features = _tmp_model.classifier[-1].out_features
        assert classifier_out_features == NUM_CLASSES, (
            f"FATAL: LateFusionModel has {classifier_out_features} output classes but "
            f"DIAGNOSTIC_CLASSES has {NUM_CLASSES}. These MUST match."
        )
        del _tmp_model
        
        self.workflow = StateGraph(AgentState)
        self._build_graph()

    def _build_graph(self):
        # Define Nodes
        self.workflow.add_node("extract_visuals", self.node_extract_visuals)
        self.workflow.add_node("parse_history", self.node_parse_history)
        self.workflow.add_node("query_pubmed", self.node_query_pubmed)
        self.workflow.add_node("synthesize_diagnosis", self.node_synthesize_diagnosis)
        self.workflow.add_node("self_verify", self.node_self_verify)
        
        # Define Edges
        self.workflow.set_entry_point("extract_visuals")
        self.workflow.add_edge("extract_visuals", "parse_history")
        self.workflow.add_edge("parse_history", "query_pubmed")
        self.workflow.add_edge("query_pubmed", "synthesize_diagnosis")
        self.workflow.add_edge("synthesize_diagnosis", "self_verify")
        
        # Conditional Edge: Self-Verification
        self.workflow.add_conditional_edges(
            "self_verify",
            self.should_continue,
            {
                "retry": "query_pubmed",
                "end": END
            }
        )
        
        self.app = self.workflow.compile()

    # Node Functions
    async def node_extract_visuals(self, state: AgentState):
        logger.info("[Node] Extracting Visuals...")
        
        orig_img_path = state['image_path']
        
        # Mask the burned-in text and retrieve the anonymized image path
        img_to_encode = self.scrubber.mask_burned_in_text(orig_img_path)
        success = (img_to_encode != orig_img_path)
        
        # Call inference API. Attempt to use shared volume path first to avoid I/O double-writing
        try:
            try:
                abs_img_path = os.path.abspath(img_to_encode)
                async with httpx.AsyncClient(verify=self.ssl_verify, cert=self.ssl_cert) as client:
                    resp = await client.post(
                        f"{self.inference_api_url}/encode/image_path",
                        json={"image_path": abs_img_path},
                        headers={"X-Internal-API-Key": self.internal_api_key},
                        timeout=10
                    )
                # If the path endpoint isn't supported or fails, fall back to upload bytes
                if resp.status_code == 404 or resp.status_code == 400:
                    raise ValueError("Endpoint failed or path not accessible. Falling back to HTTP multipart upload.")
                resp.raise_for_status()
                resp_data = resp.json()
                features = resp_data["features"]
                heatmap_base64 = resp_data.get("heatmap_base64", "")
            except Exception:
                # Fallback to standard multipart upload
                try:
                    with open(img_to_encode, "rb") as f:
                        files = {"image": ("image.jpg", f, "image/jpeg")}
                        async with httpx.AsyncClient(verify=self.ssl_verify, cert=self.ssl_cert) as client:
                            resp = await client.post(
                                f"{self.inference_api_url}/encode/image",
                                files=files,
                                headers={"X-Internal-API-Key": self.internal_api_key},
                                timeout=15
                            )
                    resp.raise_for_status()
                    resp_data = resp.json()
                    features = resp_data["features"]
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
                
        return {"visual_features": features, "heatmap_base64": heatmap_base64}

    async def node_parse_history(self, state: AgentState):
        logger.info("[Node] Parsing Patient History...")
        loop = asyncio.get_running_loop()
        raw_history = await loop.run_in_executor(None, self.parser.parse_pdf, state['patient_pdf_path'])
        
        # Scrub PHI (Fixes 'De-identification Edge Cases')
        history = await loop.run_in_executor(None, self.scrubber.scrub_history_data, raw_history)
        
        return {"history_data": history}

    async def node_query_pubmed(self, state: AgentState):
        logger.info("[Node] Querying PubMed...")
        # Use chief complaint + iteration context for refined queries
        base_query = state['history_data'].get('chief_complaint', "Chest X-ray findings")
        if state.get('iteration_count', 0) > 0:
            query = f"{base_query} differential diagnosis respiratory imaging"
        else:
            query = base_query
            
        loop = asyncio.get_running_loop()
        try:
            citations = await loop.run_in_executor(None, self.rag.search, query, 3)
            return {"pubmed_citations": citations}
        except Exception as e:
            logger.error(f"[Clinical Graph] RAG retrieval failed: {e}. Forcing human review escalation.")
            return {"pubmed_citations": [], "escalation_required": True}

    async def node_synthesize_diagnosis(self, state: AgentState):
        logger.info("[Node] Synthesizing Diagnosis...")
        # Get visual features (already list from API)
        v = state['visual_features']
            
        # Get text features from history
        history = state['history_data']
        text_content = f"{history.get('chief_complaint', '')} {history.get('history_present_illness', '')} {history.get('labs', '')}"
        
        # Embed text using remote API
        try:
            async with httpx.AsyncClient(verify=self.ssl_verify, cert=self.ssl_cert) as client:
                resp = await client.post(
                    f"{self.inference_api_url}/encode/text",
                    json={"text": text_content},
                    headers={"X-Internal-API-Key": self.internal_api_key}
                )
            resp.raise_for_status()
            t = resp.json()["embeddings"]
        except Exception as e:
            logger.error(f"[Clinical Graph] Error calling inference API text encoder: {e}")
            raise RuntimeError(f"Text encoder failed: {e}")
        
        # Run uncertainty estimation via remote API
        try:
            async with httpx.AsyncClient(verify=self.ssl_verify, cert=self.ssl_cert) as client:
                resp = await client.post(
                    f"{self.inference_api_url}/estimate",
                    json={
                        "visual_features": v,
                        "text_features": t,
                        "num_passes": 20
                    },
                    headers={"X-Internal-API-Key": self.internal_api_key}
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

        # Flaw #6 Fix: OOD detection — reject predictions where max softmax is below threshold
        ood_flag = max_softmax < OOD_CONFIDENCE_THRESHOLD
        if ood_flag:
            logger.warning(
                f"[OOD Detection] max(softmax) = {max_softmax:.4f} < threshold {OOD_CONFIDENCE_THRESHOLD}. "
                f"Input may contain pathology outside training distribution. Flagging for escalation."
            )
            top_finding = "Out-of-Distribution"
        
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
        }

    async def node_self_verify(self, state: AgentState):
        logger.info(f"[Node] Self-Verifying (Confidence: {state['confidence']:.2f}, Uncertainty: {state['diagnosis']['uncertainty_std']:.4f})...")
        count = state.get('iteration_count', 0) + 1
        
        # Flaw #3 Fix: Scale standard deviation with UNCERTAINTY_CALIBRATION_FACTOR before evaluating threshold
        scaled_std = state['diagnosis']['uncertainty_std'] * UNCERTAINTY_CALIBRATION_FACTOR
        
        # Flaw #15 Fix: Use configurable thresholds from module-level constants
        is_uncertain = state['confidence'] < CONFIDENCE_THRESHOLD or scaled_std > UNCERTAINTY_THRESHOLD
        
        # If OOD was already flagged, skip retry and go straight to escalation
        if state.get('escalation_required', False):
            logger.warning("--- OOD escalation already triggered. Ending graph. ---")
            return {"iteration_count": count, "escalation_required": True}
        
        if is_uncertain and count < MAX_RETRY_ITERATIONS:
            logger.info("--- Looping back for refined PubMed query ---")
            return {"iteration_count": count}
        
        # Escalation path: uncertainty too high or max iterations reached with uncertainty
        if is_uncertain:
            logger.warning("!!! Escalation triggered: Insufficient evidence for automated diagnosis.")
            return {"iteration_count": count, "escalation_required": True}
            
        return {"iteration_count": count}

    def should_continue(self, state: AgentState):
        if state.get('escalation_required', False):
            return "end"
        
        confidence = state.get('confidence', 1.0)
        diagnosis = state.get('diagnosis', {})
        uncertainty_std = diagnosis.get('uncertainty_std', 0.0)
        scaled_std = uncertainty_std * UNCERTAINTY_CALIBRATION_FACTOR
        
        is_uncertain = confidence < CONFIDENCE_THRESHOLD or scaled_std > UNCERTAINTY_THRESHOLD
        
        if is_uncertain and state.get('iteration_count', 0) < MAX_RETRY_ITERATIONS:
            return "retry"
        return "end"

    async def run(self, image_path: str, pdf_path: str):
        initial_state = {
            "image_path": image_path,
            "patient_pdf_path": pdf_path,
            "iteration_count": 0,
            "escalation_required": False,
            "pubmed_citations": [],
            "visual_features": None,
            "history_data": {},
            "diagnosis": {},
            "confidence": 0.0,
            "heatmap_base64": ""
        }
        return await self.app.ainvoke(initial_state)

if __name__ == "__main__":
    pass
