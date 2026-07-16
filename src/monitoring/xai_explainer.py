from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import os
import json

from src.models.fusion import DIAGNOSTIC_CLASSES


@dataclass
class EvidenceItem:
    """Structured evidence supporting a finding."""
    type: str  # "model_probability", "patient_demographics", "occupational_exposure", "clinical_symptoms", "lab_results", "literature"
    description: str
    weight: float = 1.0  # Relative importance
    source: str = ""  # Source of evidence


@dataclass
class StructuredExplanation:
    """Machine-readable structured explanation for a diagnosis."""
    diagnosis: str
    confidence: float
    uncertainty: float
    probability_distribution: Dict[str, float]
    evidence: List[EvidenceItem]
    ood_detected: bool = False
    escalation_recommended: bool = False
    rationale_summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class XAIExplainer:
    """
    Generates structured clinical rationales for AI predictions.
    Produces both machine-readable structured output and human-readable narrative.
    """
    def __init__(self):
        pass

    def explain(
        self, 
        diagnosis: str, 
        confidence: float, 
        uncertainty: float, 
        probabilities: List[float] = None, 
        history_data: Dict[str, Any] = None, 
        pubmed_citations: List[Dict[str, Any]] = None
    ) -> str:
        """Generate explanation - returns human-readable string for backwards compatibility."""
        structured = self.explain_structured(diagnosis, confidence, uncertainty, probabilities, history_data, pubmed_citations)
        return self._format_narrative(structured)

    def explain_structured(
        self, 
        diagnosis: str, 
        confidence: float, 
        uncertainty: float, 
        probabilities: List[float] = None, 
        history_data: Dict[str, Any] = None, 
        pubmed_citations: List[Dict[str, Any]] = None
    ) -> StructuredExplanation:
        """Generate structured machine-readable explanation."""
        history_data = history_data or {}
        metadata = history_data.get("metadata", {})
        pubmed_citations = pubmed_citations or []
        
        age = metadata.get("age", "Unknown")
        gender = metadata.get("gender", "Unknown")
        occupation = metadata.get("occupation", "Unknown")
        exposure_years = metadata.get("exposure_years", "0")
        chief_complaint = history_data.get("chief_complaint", "respiratory symptoms")
        hpi = history_data.get("history_present_illness", "")
        labs = history_data.get("labs", "")
        
        if chief_complaint.lower() == "not found":
            chief_complaint = "respiratory distress"

        evidence = []
        
        # Factor 1: Model probabilities and confidence
        if probabilities and len(probabilities) == len(DIAGNOSTIC_CLASSES):
            indexed_probs = list(enumerate(probabilities))
            sorted_probs = sorted(indexed_probs, key=lambda x: x[1], reverse=True)
            if len(sorted_probs) >= 2:
                top_idx, top_prob = sorted_probs[0]
                sec_idx, sec_prob = sorted_probs[1]
                gap = top_prob - sec_prob
                evidence.append(EvidenceItem(
                    type="model_probability",
                    description=f"Model classified '{DIAGNOSTIC_CLASSES[top_idx]}' with {top_prob:.1%} probability (Margin of {gap:.1%} over '{DIAGNOSTIC_CLASSES[sec_idx]}').",
                    weight=top_prob,
                    source="fusion_model"
                ))
        
        # Factor 2: Patient demographics & exposure
        demo_parts = []
        if age != "Unknown": demo_parts.append(f"{age}-year-old")
        if gender != "Unknown": demo_parts.append(f"{gender}")
        demo_text = " ".join(demo_parts) if demo_parts else "Patient"
        
        if occupation != "Unknown" or exposure_years != "0":
            evidence.append(EvidenceItem(
                type="occupational_exposure",
                description=f"{demo_text} has an occupational history as a {occupation} with {exposure_years} years of exposure.",
                weight=0.8,
                source="patient_history"
            ))
        
        # Factor 3: Clinical presentation
        if chief_complaint and chief_complaint.lower() != "not found":
            evidence.append(EvidenceItem(
                type="clinical_symptoms",
                description=f"Primary presentation is '{chief_complaint}'.",
                weight=0.9,
                source="patient_history"
            ))

        # Factor 4: HPI symptoms
        if hpi:
            symptoms = []
            for sym in ["cough", "dyspnea", "fever", "sputum", "weight loss", "night sweats", "chest pain", "shortness of breath", "hemoptysis", "wheezing"]:
                if sym in hpi.lower() or sym in chief_complaint.lower():
                    symptoms.append(sym)
            if symptoms:
                evidence.append(EvidenceItem(
                    type="clinical_symptoms",
                    description=f"History of present illness includes: {', '.join(symptoms)}.",
                    weight=0.85,
                    source="patient_history"
                ))
        
        # Factor 5: Labs
        if labs and labs.lower() not in ["none", "unknown", ""]:
            evidence.append(EvidenceItem(
                type="lab_results",
                description=f"Laboratory findings: {labs.strip()}.",
                weight=0.7,
                source="patient_history"
            ))

        # Factor 6: Literature support
        if pubmed_citations:
            citations_list = []
            for cit in pubmed_citations:
                title = cit.get("title", "Unknown Title")
                pmid = cit.get("pmid", "N/A")
                citations_list.append(f"'{title}' (PMID: {pmid})")
            if citations_list:
                evidence.append(EvidenceItem(
                    type="literature",
                    description="Supporting PubMed Literature: " + "; ".join(citations_list) + ".",
                    weight=0.6,
                    source="pubmed"
                ))

        # Build probability distribution
        prob_dict = {}
        if probabilities and len(probabilities) == len(DIAGNOSTIC_CLASSES):
            for i, prob in enumerate(probabilities):
                prob_dict[DIAGNOSTIC_CLASSES[i]] = float(prob)

        # OOD / Uncertainty flags
        uncertainty_threshold = float(os.getenv("UNCERTAINTY_THRESHOLD", "0.15"))
        ood_detected = diagnosis == "Out-of-Distribution"
        escalation_recommended = ood_detected or uncertainty > uncertainty_threshold or confidence < 0.6

        # Generate narrative summary
        rationale = self._build_rationale(evidence, diagnosis, confidence, uncertainty, ood_detected, escalation_recommended)

        return StructuredExplanation(
            diagnosis=diagnosis,
            confidence=confidence,
            uncertainty=uncertainty,
            probability_distribution=prob_dict,
            evidence=evidence,
            ood_detected=ood_detected,
            escalation_recommended=escalation_recommended,
            rationale_summary=rationale
        )

    def _build_rationale(self, evidence: List[EvidenceItem], diagnosis: str, 
                         confidence: float, uncertainty: float, 
                         ood_detected: bool, escalation: bool) -> str:
        """Build human-readable rationale summary."""
        parts = []
        
        if ood_detected:
            parts.append(f"CAUTION: Out-of-distribution detection triggered. Unknown pathology suspected.")
        elif escalation:
            parts.append(f"CAUTION: High predictive uncertainty ({uncertainty:.3f}) or low confidence ({confidence:.1%}) suggests non-standard presentation.")
        
        parts.append(f"Diagnosis: {diagnosis} with {confidence:.1%} confidence (±{uncertainty:.3f} std dev from MC Dropout).")
        
        for ev in evidence:
            parts.append(ev.description)
        
        return " ".join(parts)

    def _format_narrative(self, explanation: StructuredExplanation) -> str:
        """Format structured explanation as human-readable narrative (backwards compatible)."""
        return explanation.rationale_summary


if __name__ == "__main__":
    explainer = XAIExplainer()
    result = explainer.explain_structured(
        diagnosis="Silicosis",
        confidence=0.72,
        uncertainty=0.08,
        probabilities=[0.72, 0.10, 0.05, 0.08, 0.05],
        history_data={
            "chief_complaint": "progressive dyspnea",
            "history_present_illness": "2-year history of progressive dyspnea on exertion and dry cough",
            "metadata": {"age": "58", "gender": "Male", "occupation": "sandblaster", "exposure_years": "25"}
        },
        pubmed_citations=[{"title": "Silicosis in Sandblasters", "pmid": "12345678"}]
    )
    print("Structured:")
    print(result.to_json())
    print("\nNarrative:")
    print(result.rationale_summary)