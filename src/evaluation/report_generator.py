from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from datetime import datetime
import os

class ClinicalReportGenerator:
    """
    Generates professional PDF clinical reports for diagnostic results.
    Includes visual heatmaps, RAG citations, and uncertainty quantification.
    """
    def __init__(self, output_dir="outputs/reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()

    def generate_report(self, diagnosis_result, patient_metadata, heatmap_path, citations, output_filename=None):
        if not output_filename:
            pid = patient_metadata.get('patient_id', 'UNK')
            date_str = datetime.now().strftime("%Y%m%d_%H%M")
            output_filename = f"Report_{pid}_{date_str}.pdf"
        
        filepath = os.path.join(self.output_dir, output_filename)
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        elements = []

        # Header
        elements.append(Paragraph("MEdi Chain AI - Diagnostic Report", self.styles['Title']))
        elements.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y %H:%M')}", self.styles['Normal']))
        elements.append(Spacer(1, 12))

        # Patient Info
        elements.append(Paragraph("Patient Information", self.styles['Heading2']))
        patient_data = [
            ["Patient ID:", patient_metadata.get('patient_id', 'Unknown')],
            ["Age:", str(patient_metadata.get('age', 'Unknown'))],
            ["Gender:", patient_metadata.get('gender', 'Unknown')],
            ["Occupation:", patient_metadata.get('occupation', 'Unknown')],
        ]
        t = Table(patient_data, colWidths=[100, 300])
        t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
        elements.append(t)
        elements.append(Spacer(1, 12))

        # Diagnostic Summary
        elements.append(Paragraph("Diagnostic Interpretation", self.styles['Heading2']))
        findings = diagnosis_result.get('top_finding', 'Inconclusive')
        confidence = diagnosis_result.get('confidence', 0.0)
        uncertainty = diagnosis_result.get('uncertainty_std', 0.0)
        
        summary_text = f"<b>Primary Finding:</b> {findings}<br/>"
        summary_text += f"<b>Confidence:</b> {confidence:.1%} (±{uncertainty:.1%})<br/>"
        
        if diagnosis_result.get('escalation_required', False):
            summary_text += "<br/><font color='red'><b>URGENT: Insufficient evidence for automated diagnosis. Escalated to Radiologist.</b></font>"
        
        elements.append(Paragraph(summary_text, self.styles['Normal']))
        elements.append(Spacer(1, 12))

        # Visual Evidence (Heatmap)
        if os.path.exists(heatmap_path):
            elements.append(Paragraph("Visual Evidence (Grad-CAM Heatmap)", self.styles['Heading3']))
            img = Image(heatmap_path, width=300, height=300)
            elements.append(img)
            elements.append(Paragraph("<i>Red regions indicates anatomical areas used for clinical classification.</i>", self.styles['Italic']))
            elements.append(Spacer(1, 12))

        # RAG Citations
        if citations:
            elements.append(Paragraph("Biomedical Literatures / Citations", self.styles['Heading3']))
            for cit in citations:
                # cit is a dict: {'pmid': ..., 'text': ..., 'title': ...}
                title = cit.get('title', 'Unknown Title')
                pmid = cit.get('pmid', 'N/A')
                snippet = cit.get('text', '')[:200] + "..."
                elements.append(Paragraph(f"<b>{title}</b> (PMID: {pmid})", self.styles['Normal']))
                elements.append(Paragraph(snippet, self.styles['Italic']))
                elements.append(Spacer(1, 6))

        # Differential Diagnosis Table
        elements.append(Paragraph("Differential Diagnosis Probabilities", self.styles['Heading3']))
        probs = diagnosis_result.get('probabilities', [])
        classes = ["Silicosis", "Pneumonia", "Tuberculosis", "Asbestosis", "Normal"]
        diff_data = [["Condition", "Probability"]]
        for i, p in enumerate(probs):
            if i < len(classes):
                diff_data.append([classes[i], f"{p:.1%}"])
        
        dt = Table(diff_data, colWidths=[150, 100])
        dt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
        ]))
        elements.append(dt)

        # Build PDF
        doc.build(elements)
        print(f"Report generated successfully: {filepath}")
        return filepath

if __name__ == "__main__":
    # Test
    gen = ClinicalReportGenerator()
    res = {
        "top_finding": "Silicosis",
        "confidence": 0.72,
        "uncertainty_std": 0.08,
        "probabilities": [0.72, 0.18, 0.05, 0.03, 0.02],
        "escalation_required": False
    }
    meta = {"patient_id": "P999", "age": 54, "gender": "Male", "occupation": "Driller"}
    citations = [{"title": "Silicosis in construction workers", "pmid": "12345", "text": "Extensive study on dust exposure..."}]
    
    # Try to use the heatmap we generated earlier
    h_path = "outputs/heatmaps/sample_heatmap.png"
    gen.generate_report(res, meta, h_path, citations)
