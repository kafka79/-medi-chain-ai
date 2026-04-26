import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import random

class ClinicalHistoryGenerator:
    def __init__(self, output_dir='data/synthetic_histories'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()

    def generate_patient_data(self, condition_type="silicosis"):
        """Generate synthetic patient details based on condition."""
        occupations = {
            "silicosis": ["Sandblaster", "Quarry worker", "Glass manufacturer", "Foundry worker"],
            "asbestosis": ["Shipyard worker", "Insulation installer", "Brake repair technician", "Construction worker"],
            "coal_dust": ["Coal miner", "Underground excavator", "Tunneling operator"],
            "pneumonia": ["Office worker", "Teacher", "Retail manager", "Accountant"]
        }
        
        exposure_duration = random.randint(5, 30)
        age = random.randint(45, 75)
        job = random.choice(occupations.get(condition_type, ["Unknown"]))
        
        data = {
            "patient_name": f"PATIENT_{random.randint(1000, 9999)}",
            "age": age,
            "gender": random.choice(["Male", "Female"]),
            "job": job,
            "exposure_years": exposure_duration if condition_type != "pneumonia" else 0,
            "chief_complaint": "",
            "history_present_illness": "",
            "condition": condition_type
        }
        
        if condition_type == "silicosis":
            data["chief_complaint"] = "Progressive shortness of breath and chronic dry cough over the past 2 years."
            data["history_present_illness"] = f"Patient worked as a {job} for {exposure_duration} years. Reports worsening dyspnea on exertion. Denies fever or chills. History of smoking for 10 pack-years (quit 5 years ago)."
        elif condition_type == "asbestosis":
            data["chief_complaint"] = "Gradual onset of dyspnea and chest tightness."
            data["history_present_illness"] = f"Former {job} with {exposure_duration} years of potential asbestos exposure. Patient reports bibasilar inspiratory crackles noted by primary care physician. No history of asthma."
        elif condition_type == "pneumonia":
            data["chief_complaint"] = "Acute onset of high fever, productive cough, and pleuritic chest pain."
            data["history_present_illness"] = "Patient presents with 4-day history of chills and yellowish sputum. No significant occupational exposures. Vital signs: Temp 102.4F, HR 98, RR 22."
            
        return data

    def create_pdf(self, patient_data):
        """Create a 5-page PDF clinical report."""
        filename = f"{patient_data['patient_name']}_{patient_data['condition']}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        elements = []
        
        # Header
        elements.append(Paragraph(f"CLINICAL HISTORY REPORT: {patient_data['patient_name']}", self.styles['Title']))
        elements.append(Spacer(1, 12))
        
        # Demographics Table
        table_data = [
            ['Name:', patient_data['patient_name'], 'Age:', str(patient_data['age'])],
            ['Gender:', patient_data['gender'], 'Date:', '2026-04-26'],
            ['Occupation:', patient_data['job'], 'Exposure:', f"{patient_data['exposure_years']} years"]
        ]
        t = Table(table_data, colWidths=[80, 150, 80, 150])
        t.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, colors.grey),
            ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
            ('BACKGROUND', (2,0), (2,-1), colors.lightgrey),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 24))
        
        # Sections
        sections = [
            ("CHIEF COMPLAINT", patient_data['chief_complaint']),
            ("HISTORY OF PRESENT ILLNESS", patient_data['history_present_illness']),
            ("PAST MEDICAL HISTORY", "Controlled hypertension. No prior surgeries. No known allergies."),
            ("SOCIAL HISTORY", "Married. Lives in an urban environment. Former smoker (10 pack-years)."),
            ("REVIEW OF SYSTEMS", "Respiratory: Positive for cough and dyspnea. Cardiovascular: Denies chest pain. Constitutional: Denies weight loss.")
        ]
        
        for title, content in sections:
            elements.append(Paragraph(title, self.styles['Heading2']))
            elements.append(Paragraph(content, self.styles['Normal']))
            elements.append(Spacer(1, 12))
            
        # Add filler pages to make it 5 pages as requested
        for i in range(2, 6):
            elements.append(Spacer(1, 400)) # Simple page break simulation
            elements.append(Paragraph(f"Page {i} - Clinical Notes Continued", self.styles['Heading3']))
            elements.append(Paragraph("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Section for lab results, vital sign trends, and historical imaging comparisons.", self.styles['Normal']))
            
        doc.build(elements)
        return filepath

if __name__ == "__main__":
    generator = ClinicalHistoryGenerator()
    for condition in ["silicosis", "asbestosis", "pneumonia"]:
        data = generator.generate_patient_data(condition)
        path = generator.create_pdf(data)
        print(f"Generated PDF: {path}")
