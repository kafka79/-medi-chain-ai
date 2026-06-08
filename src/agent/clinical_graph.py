from typing import TypedDict, List, Dict, Any, Union
from langgraph.graph import StateGraph, END
import torch
import numpy as np

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

class ClinicalAgent:
    def __init__(self, history_parser, rag_evaluator, inference_api_url: str = None):
        self.parser = history_parser
        self.rag = rag_evaluator
        import os
        self.inference_api_url = inference_api_url or os.getenv("INFERENCE_API_URL", "http://inference-api:8001")
        self.internal_api_key = os.getenv("INTERNAL_API_KEY", "internal-secret-token")
        
        from ..data.privacy_scrubber import PrivacyScrubber
        self.scrubber = PrivacyScrubber()
        
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
    def node_extract_visuals(self, state: AgentState):
        print("[Node] Extracting Visuals...")
        import os
        from pathlib import Path
        
        orig_img_path = state['image_path']
        
        # Mask the burned-in text and retrieve the anonymized image path
        img_to_encode = self.scrubber.mask_burned_in_text(orig_img_path)
        success = (img_to_encode != orig_img_path)
        
        import requests
        
        # Call inference API instead of local encoder
        try:
            with open(img_to_encode, "rb") as f:
                resp = requests.post(
                    f"{self.inference_api_url}/encode/image",
                    files={"image": ("image.jpg", f, "image/jpeg")},
                    headers={"X-Internal-API-Key": self.internal_api_key}
                )
            resp.raise_for_status()
            features = resp.json()["features"]
        except Exception as e:
            print(f"[Clinical Graph] Error calling inference API: {e}")
            raise RuntimeError(f"Visual encoder failed: {e}")
        
        # Clean up the temporary scrubbed image if it was created
        if success and os.path.exists(img_to_encode) and img_to_encode != orig_img_path:
            try:
                os.remove(img_to_encode)
            except Exception as e:
                print(f"[Clinical Graph] Warning: failed to clean up temp scrubbed image: {e}")
                
        return {"visual_features": features}

    def node_parse_history(self, state: AgentState):
        print("[Node] Parsing Patient History...")
        raw_history = self.parser.parse_pdf(state['patient_pdf_path'])
        
        # Scrub PHI (Fixes 'De-identification Edge Cases')
        history = self.scrubber.scrub_history_data(raw_history)
        
        return {"history_data": history}

    def node_query_pubmed(self, state: AgentState):
        print("[Node] Querying PubMed...")
        # Use chief complaint + iteration context for refined queries
        base_query = state['history_data'].get('chief_complaint', "Chest X-ray findings")
        if state.get('iteration_count', 0) > 0:
            query = f"{base_query} differential diagnosis respiratory imaging"
        else:
            query = base_query
            
        citations = self.rag.search(query, k=3)
        return {"pubmed_citations": citations}

    def node_synthesize_diagnosis(self, state: AgentState):
        print("[Node] Synthesizing Diagnosis...")
        # Get visual features (already list from API)
        v = state['visual_features']
            
        # Get text features from history
        history = state['history_data']
        text_content = f"{history.get('chief_complaint', '')} {history.get('history_present_illness', '')} {history.get('labs', '')}"
        
        import requests
        
        # Embed text using remote API
        try:
            resp = requests.post(
                f"{self.inference_api_url}/encode/text",
                json={"text": text_content},
                headers={"X-Internal-API-Key": self.internal_api_key}
            )
            resp.raise_for_status()
            t = resp.json()["embeddings"]
        except Exception as e:
            print(f"[Clinical Graph] Error calling inference API text encoder: {e}")
            raise RuntimeError(f"Text encoder failed: {e}")
        
        # Run uncertainty estimation via remote API
        try:
            resp = requests.post(
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
            print(f"[Clinical Graph] Error calling inference API uncertainty estimator: {e}")
            raise RuntimeError(f"Uncertainty estimator failed: {e}")
        
        pred_idx = int(results['prediction'][0])
        classes = ["Silicosis", "Pneumonia", "Tuberculosis", "Asbestosis", "Normal"]
        top_finding = classes[pred_idx] if pred_idx < len(classes) else "Unknown"
        
        # Generate Clinical Rationale (Fixes 'User-Led Explanation')
        from src.monitoring.xai_explainer import XAIExplainer
        explainer = XAIExplainer()
        rationale = explainer.explain(
            top_finding, 
            float(results['mean_confidence'][0]),
            float(results['std_deviation'][0]),
            probabilities=results['all_probs'][0],
            history_data=history
        )
        
        diagnosis = {
            "top_finding": top_finding,
            "rationale": rationale,
            "probabilities": results['all_probs'][0],
            "uncertainty_std": float(results['std_deviation'][0])
        }
        
        return {
            "diagnosis": diagnosis, 
            "confidence": float(results['mean_confidence'][0])
        }

    def node_self_verify(self, state: AgentState):
        print(f"[Node] Self-Verifying (Confidence: {state['confidence']:.2f}, Uncertainty: {state['diagnosis']['uncertainty_std']:.4f})...")
        count = state.get('iteration_count', 0) + 1
        
        # Thresholds per plan: confidence < 0.6 or uncertainty_std > 0.15
        is_uncertain = state['confidence'] < 0.6 or state['diagnosis']['uncertainty_std'] > 0.15
        
        if is_uncertain and count < 3:
            print("--- Looping back for refined PubMed query ---")
            return {"iteration_count": count}
        
        # Escalation path: uncertainty too high or max iterations reached with uncertainty
        if is_uncertain:
            print("!!! Escalation triggered: Insufficient evidence for automated diagnosis.")
            return {"iteration_count": count, "escalation_required": True}
            
        return {"iteration_count": count}

    def should_continue(self, state: AgentState):
        if state.get('escalation_required', False):
            return "end"
        
        confidence = state.get('confidence', 1.0)
        diagnosis = state.get('diagnosis', {})
        uncertainty_std = diagnosis.get('uncertainty_std', 0.0)
        
        is_uncertain = confidence < 0.6 or uncertainty_std > 0.15
        
        if is_uncertain and state.get('iteration_count', 0) < 3:
            return "retry"
        return "end"

    def run(self, image_path: str, pdf_path: str):
        initial_state = {
            "image_path": image_path,
            "patient_pdf_path": pdf_path,
            "iteration_count": 0,
            "escalation_required": False,
            "pubmed_citations": [],
            "visual_features": None,
            "history_data": {},
            "diagnosis": {},
            "confidence": 0.0
        }
        return self.app.invoke(initial_state)

if __name__ == "__main__":
    pass
