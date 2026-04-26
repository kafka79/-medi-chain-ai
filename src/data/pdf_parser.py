import pdfplumber
import re
import json
import os

class ClinicalPDFParser:
    def __init__(self):
        # Regex patterns for clinical sections
        self.patterns = {
            "chief_complaint": r"CHIEF COMPLAINT\s*(.*?)(?=\n[A-Z ]{5,}|\Z)",
            "history_present_illness": r"HISTORY OF PRESENT ILLNESS\s*(.*?)(?=\n[A-Z ]{5,}|\Z)",
            "past_medical_history": r"PAST MEDICAL HISTORY\s*(.*?)(?=\n[A-Z ]{5,}|\Z)",
            "social_history": r"SOCIAL HISTORY\s*(.*?)(?=\n[A-Z ]{5,}|\Z)",
            "review_of_systems": r"REVIEW OF SYSTEMS\s*(.*?)(?=\n[A-Z ]{5,}|\Z)",
            "labs": r"LABORATORY RESULTS\s*(.*?)(?=\n[A-Z ]{5,}|\Z)"
        }

    def parse_pdf(self, pdf_path):
        """Extract sections from a patient PDF."""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        
        # Normalize text (remove extra whitespaces)
        text = re.sub(r' +', ' ', text)
        
        extracted_data = {}
        for section, pattern in self.patterns.items():
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                extracted_data[section] = match.group(1).strip()
            else:
                extracted_data[section] = "Not found"
        
        # Extract metadata (Age, Gender, Occupation, Exposure)
        age_match = re.search(r"Age:\s*(\d+)", text)
        gender_match = re.search(r"Gender:\s*(\w+)", text)
        job_match = re.search(r"Occupation:\s*(.*?)\s*Exposure:", text)
        exposure_match = re.search(r"Exposure:\s*(\d+)\s*years", text)
        
        extracted_data["metadata"] = {
            "age": age_match.group(1) if age_match else "Unknown",
            "gender": gender_match.group(1) if gender_match else "Unknown",
            "occupation": job_match.group(1).strip() if job_match else "Unknown",
            "exposure_years": exposure_match.group(1) if exposure_match else "0"
        }
        
        return extracted_data

    def save_json(self, data, output_path):
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = ClinicalPDFParser()
    # Example usage:
    # data = parser.parse_pdf("data/synthetic_histories/PATIENT_1234_silicosis.pdf")
    # print(json.dumps(data, indent=2))
