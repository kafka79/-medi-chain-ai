from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from datetime import datetime
import os

from src.data.fhir_formatter import FHIRFormatter

class ClinicalReportGenerator:
    """
    Generates professional PDF clinical reports for diagnostic results.
    Includes visual heatmaps, RAG citations, and uncertainty quantification.
    """
    def __init__(self, output_dir="outputs/reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self.fhir_formatter = FHIRFormatter()

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

        # Clinical Key Takeaways (New High-Priority Section)
        elements.append(Paragraph("Clinical Key Takeaways", self.styles['Heading2']))
        findings = diagnosis_result.get('top_finding', 'Inconclusive')
        confidence = diagnosis_result.get('confidence', 0.0)
        uncertainty = diagnosis_result.get('uncertainty_std', 0.0)
        
        takeaway_style = ParagraphStyle(
            'Takeaway',
            parent=self.styles['Normal'],
            backColor=colors.whitesmoke,
            borderPadding=10,
            leading=14
        )
        
        takeaway_text = f"<b>Primary Finding:</b> {findings}<br/>"
        takeaway_text += f"<b>Statistical Confidence:</b> {confidence:.1%} (±{uncertainty:.1%})<br/>"
        
        if diagnosis_result.get('escalation_required', False):
            takeaway_text += "<br/><font color='red'><b>ACTION REQUIRED:</b> This case has been escalated for manual review due to insufficient automated evidence.</font>"
        else:
            takeaway_text += "<br/><b>Recommendation:</b> Correlate with clinical symptoms and history."

        elements.append(Paragraph(takeaway_text, takeaway_style))
        elements.append(Spacer(1, 12))

        # Visual Evidence
        if os.path.exists(heatmap_path):
            elements.append(Paragraph("Visual Evidence (Attention Mapping)", self.styles['Heading3']))
            img = Image(heatmap_path, width=250, height=250)
            elements.append(img)
            elements.append(Paragraph("<i>Highlighted regions indicate anatomical areas driving the AI prediction.</i>", self.styles['Italic']))
            elements.append(Spacer(1, 12))

        # Differential Diagnosis Table
        elements.append(Paragraph("Differential Diagnosis Summary", self.styles['Heading3']))
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
        elements.append(Spacer(1, 12))

        # Appendix: Supporting Literature (Condensed)
        if citations:
            elements.append(Paragraph("Appendix: Supporting Literature", self.styles['Heading3']))
            for cit in citations:
                title = cit.get('title', 'Unknown Title')
                pmid = cit.get('pmid', 'N/A')
                elements.append(Paragraph(f"• <b>{title}</b> (PMID: {pmid})", self.styles['Normal']))
            elements.append(Paragraph("<i>Full snippets available in the digital FHIR sidecar.</i>", self.styles['Italic']))

        # Build PDF
        doc.build(elements)
        fhir_path = self._write_fhir_sidecar(filepath, diagnosis_result, patient_metadata)
        print(f"Report generated successfully: {filepath}")
        return {"pdf_path": filepath, "fhir_path": fhir_path}

    def _write_fhir_sidecar(self, pdf_path, diagnosis_result, patient_metadata):
        probabilities = diagnosis_result.get('probabilities', [])
        classes = ["Silicosis", "Pneumonia", "Tuberculosis", "Asbestosis", "Normal"]
        differential = {
            classes[i]: probabilities[i]
            for i in range(min(len(classes), len(probabilities)))
        }
        diagnosis_data = {
            "patient_id": patient_metadata.get('patient_id', 'UNK'),
            "primary_finding": diagnosis_result.get('top_finding', 'Inconclusive'),
            "differential": differential,
            "confidence": diagnosis_result.get('confidence', 0.0),
            "report_path": pdf_path,
            "escalation_required": diagnosis_result.get('escalation_required', False),
        }
        report = self.fhir_formatter.create_diagnostic_report(diagnosis_data)
        fhir_path = os.path.splitext(pdf_path)[0] + ".fhir.json"
        with open(fhir_path, "w", encoding="utf-8") as handle:
            handle.write(self.fhir_formatter.to_json(report))
        return fhir_path

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
