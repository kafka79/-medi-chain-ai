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
    def __init__(self, visual_encoder, history_parser, rag_evaluator, fusion_model, text_encoder, uncertainty_estimator):
        self.encoder = visual_encoder
        self.parser = history_parser
        self.rag = rag_evaluator
        self.fusion = fusion_model
        self.text_encoder = text_encoder
        self.uncertainty = uncertainty_estimator
        
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
        features = self.encoder.encode_image(state['image_path'])
        return {"visual_features": features}

    def node_parse_history(self, state: AgentState):
        print("[Node] Parsing Patient History...")
        history = self.parser.parse_pdf(state['patient_pdf_path'])
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
        # Get visual features
        v = state['visual_features']
        if not torch.is_tensor(v):
            v = torch.from_numpy(v).float()
        if v.ndim == 1:
            v = v.unsqueeze(0)
            
        # Get text features from history
        history = state['history_data']
        text_content = f"{history.get('chief_complaint', '')} {history.get('history_present_illness', '')} {history.get('labs', '')}"
        
        # Embed text using SapBERT/PubMedBERT
        t = self.text_encoder.encode([text_content], convert_to_tensor=True)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        
        # Run uncertainty estimation (includes fusion model call)
        results = self.uncertainty.estimate_uncertainty(v, t, num_passes=20)
        
        pred_idx = results['prediction'][0].item()
        classes = ["Silicosis", "Pneumonia", "Tuberculosis", "Asbestosis", "Normal"]
        
        diagnosis = {
            "top_finding": classes[pred_idx] if pred_idx < len(classes) else "Unknown",
            "probabilities": results['all_probs'][0].tolist(),
            "uncertainty_std": results['std_deviation'][0]
        }
        
        return {
            "diagnosis": diagnosis, 
            "confidence": results['mean_confidence'][0].item()
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
        if state['confidence'] < 0.6 and state['iteration_count'] < 3:
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
