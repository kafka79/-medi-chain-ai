class XAIExplainer:
    """
    Addresses 'User-Led Explanation'.
    Generates dynamic clinical rationales for AI predictions to improve clinician trust
    by synthesizing patient demographics, occupational exposure, labs, active symptoms,
    and supporting medical literature.
    """
    def __init__(self):
        pass

    def explain(self, diagnosis: str, confidence: float, uncertainty: float, probabilities: list = None, history_data: dict = None, pubmed_citations: list = None) -> str:
        # Extract patient metadata and history fields dynamically
        history_data = history_data or {}
        metadata = history_data.get("metadata", {})
        
        age = metadata.get("age", "Unknown")
        gender = metadata.get("gender", "Unknown")
        occupation = metadata.get("occupation", "Unknown")
        exposure_years = metadata.get("exposure_years", "0")
        chief_complaint = history_data.get("chief_complaint", "respiratory symptoms")
        hpi = history_data.get("history_present_illness", "")
        labs = history_data.get("labs", "")
        
        if chief_complaint.lower() == "not found":
            chief_complaint = "respiratory distress"

        try:
            from src.models.fusion import DIAGNOSTIC_CLASSES
        except ImportError:
            DIAGNOSTIC_CLASSES = ["Silicosis", "Pneumonia", "Tuberculosis", "Asbestosis", "Normal"]

        # Dynamically build base reasoning based on feature synthesis rather than rigid rule mapping
        evidence_points = []
        
        # Factor 1: Probabilities and Confidence
        if probabilities and len(probabilities) == len(DIAGNOSTIC_CLASSES):
            indexed_probs = list(enumerate(probabilities))
            sorted_probs = sorted(indexed_probs, key=lambda x: x[1], reverse=True)
            if len(sorted_probs) >= 2:
                top_idx, top_prob = sorted_probs[0]
                sec_idx, sec_prob = sorted_probs[1]
                gap = top_prob - sec_prob
                evidence_points.append(f"Model classified '{DIAGNOSTIC_CLASSES[top_idx]}' with {top_prob:.1%} probability (Margin of {gap:.1%} over '{DIAGNOSTIC_CLASSES[sec_idx]}').")
        
        # Factor 2: Patient Demographics & Exposure
        demo_str = []
        if age != "Unknown": demo_str.append(f"{age}-year-old")
        if gender != "Unknown": demo_str.append(f"{gender}")
        demo_text = " ".join(demo_str) if demo_str else "Patient"
        
        if occupation != "Unknown" or exposure_years != "0":
            evidence_points.append(f"{demo_text} has an occupational history as a {occupation} with {exposure_years} years of exposure.")
            
        # Factor 3: Clinical Presentation
        if chief_complaint and chief_complaint.lower() != "not found":
            evidence_points.append(f"Primary presentation is '{chief_complaint}'.")

        base_reason = " ".join(evidence_points)
        
        dyn_text = f"Diagnosis of {diagnosis} with {confidence:.1%} overall confidence (±{uncertainty:.3f} std dev from MC Dropout)."
            
        import os
        uncertainty_threshold = float(os.getenv("UNCERTAINTY_THRESHOLD", "0.15"))
        
        # 2. Dynamic clinical text synthesis from HPI and Labs
        clinical_evidence = []
        if hpi:
            symptoms = []
            for sym in ["cough", "dyspnea", "fever", "sputum", "weight loss", "night sweats", "chest pain", "shortness of breath"]:
                if sym in hpi.lower() or sym in chief_complaint.lower():
                    symptoms.append(sym)
            if symptoms:
                clinical_evidence.append(f"clinical symptoms of {', '.join(symptoms)}")
        if labs and labs.lower() not in ["none", "unknown", ""]:
            clinical_evidence.append(f"labs: {labs.strip()}")
            
        evidence_synthesis = ""
        if clinical_evidence:
            evidence_synthesis = " Patient history corroborates this with " + " and ".join(clinical_evidence) + "."

        # 3. Dynamic RAG/PubMed Literature context
        citation_info = ""
        if pubmed_citations:
            citations_list = []
            for cit in pubmed_citations:
                title = cit.get("title", "Unknown Title")
                pmid = cit.get("pmid", "N/A")
                citations_list.append(f"'{title}' (PMID: {pmid})")
            if citations_list:
                citation_info = " Supporting PubMed Literature: " + "; ".join(citations_list) + "."

        if diagnosis == "Out-of-Distribution":
            return f"CAUTION: {dyn_text} {base_reason} Unknown pathology detected. Immediate manual clinical review is required.{evidence_synthesis}{citation_info}"
            
        if uncertainty > uncertainty_threshold:
            return f"CAUTION: {dyn_text} {base_reason} High predictive uncertainty suggests a non-standard presentation. Manual review of priors is advised.{evidence_synthesis}{citation_info}"
        
        return f"Rationale: {dyn_text} {base_reason}{evidence_synthesis}{citation_info}"
