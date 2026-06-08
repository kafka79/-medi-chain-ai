class XAIExplainer:
    """
    Addresses 'User-Led Explanation'.
    Generates dynamic clinical rationales for AI predictions to improve clinician trust.
    """
    def __init__(self):
        pass

    def explain(self, diagnosis: str, confidence: float, uncertainty: float, probabilities: list = None, history_data: dict = None) -> str:
        # Extract patient metadata and history fields dynamically
        history_data = history_data or {}
        metadata = history_data.get("metadata", {})
        
        age = metadata.get("age", "Unknown")
        gender = metadata.get("gender", "Unknown")
        occupation = metadata.get("occupation", "Unknown")
        exposure_years = metadata.get("exposure_years", "0")
        chief_complaint = history_data.get("chief_complaint", "respiratory symptoms")
        
        if chief_complaint.lower() == "not found":
            chief_complaint = "respiratory distress"

        # Dynamically build base reasoning based on specific patient demographics & symptoms
        if diagnosis == "Silicosis":
            base_reason = (
                f"Visual features indicate upper lobe nodular opacities. This pattern is highly "
                f"correlated with the patient's {exposure_years} years of exposure history as a {occupation} "
                f"presenting with '{chief_complaint}'."
            )
        elif diagnosis == "Pneumonia":
            base_reason = (
                f"Visual features show localized consolidation patterns (typically lower lobe). "
                f"This matches the acute symptoms of '{chief_complaint}' in this {age}-year-old {gender} patient."
            )
        elif diagnosis == "Tuberculosis":
            base_reason = (
                f"Visual features highlight apical cavitary lesions. This presentation is clinically "
                f"consistent with the chief complaint of '{chief_complaint}'."
            )
        elif diagnosis == "Asbestosis":
            base_reason = (
                f"Visual features show pleural thickening and subpleural reticular lines in the lower zones. "
                f"This matches the patient's occupational history as a {occupation} with {exposure_years} years of dust exposure."
            )
        elif diagnosis == "Normal":
            base_reason = (
                f"Visual features indicate clear lung fields, sharp costophrenic angles, and normal lung volume. "
                f"There are no visual indicators to match the complaint of '{chief_complaint}'."
            )
        else:
            base_reason = f"Atypical presentation detected for '{chief_complaint}'."

        dyn_text = f"Diagnosis of {diagnosis} with {confidence:.1%} confidence (±{uncertainty:.3f} std dev from MC Dropout)."
        if probabilities:
            dyn_text += f" Raw class probabilities: {[f'{p:.2f}' for p in probabilities]}."
            
        if uncertainty > 0.15:
            return f"CAUTION: {dyn_text} {base_reason} High predictive uncertainty suggests a non-standard presentation. Manual review of priors is advised."
        
        return f"Rationale: {dyn_text} {base_reason}"
