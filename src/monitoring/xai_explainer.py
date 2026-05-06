class XAIExplainer:
    """
    Addresses 'User-Led Explanation'.
    Generates clinical rationales for AI predictions to improve clinician trust.
    """
    def __init__(self):
        self.reasoning_templates = {
            "Silicosis": "High visual attention on upper lobe nodular opacities combined with recorded history of dust exposure (e.g., mining/drilling).",
            "Pneumonia": "Consolidation patterns detected in lower lobes with acute symptoms (fever/cough) in clinical history.",
            "Tuberculosis": "Cavitary lesions in apical regions with persistent cough and weight loss reported in history.",
            "Asbestosis": "Pleural thickening and lower zone reticular patterns with chronic exposure history.",
            "Normal": "Clear lung fields, sharp costophrenic angles, and no significant history of respiratory distress."
        }

    def explain(self, diagnosis: str, confidence: float, uncertainty: float) -> str:
        base_reason = self.reasoning_templates.get(diagnosis, "Atypical pattern detected.")
        
        if uncertainty > 0.12:
            return f"CAUTION: {base_reason} However, high predictive uncertainty suggests a non-standard presentation. Manual review of priors is advised."
        
        return f"Rationale: {base_reason} (AI Confidence: {confidence:.1%})"
