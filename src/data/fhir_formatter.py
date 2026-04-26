from fhir.resources.diagnosticreport import DiagnosticReport
from fhir.resources.identifier import Identifier
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.reference import Reference
import json
from datetime import datetime

class FHIRFormatter:
    def __init__(self):
        pass

    def create_diagnostic_report(self, diagnosis_data):
        """
        Create a FHIR-R4 DiagnosticReport from raw diagnosis results.
        diagnosis_data: dict with keys: patient_id, differential, confidence, findings
        """
        report = DiagnosticReport.construct()
        
        # Identifier
        report.identifier = [Identifier(value=f"RPT-{diagnosis_data.get('patient_id', 'UNK')}-{int(datetime.now().timestamp())}")]
        
        # Status
        report.status = "final"
        
        # Category (LOINC code for Radiology Report)
        report.category = [CodeableConcept(coding=[Coding(
            system="http://terminology.hl7.org/CodeSystem/v2-0074",
            code="RAD",
            display="Radiology"
        )])]
        
        # Code (LOINC code for Chest X-ray)
        report.code = CodeableConcept(coding=[Coding(
            system="http://loinc.org",
            code="18748-4",
            display="Diagnostic imaging report"
        )])
        
        # Subject
        report.subject = Reference(display=f"Patient {diagnosis_data.get('patient_id', 'Unknown')}")
        
        # Effective DateTime
        report.effectiveDateTime = datetime.now().isoformat()
        
        # Conclusion / Differential Diagnosis
        differential_str = ", ".join([f"{k}: {v:.1%}" for k, v in diagnosis_data.get('differential', {}).items()])
        report.conclusion = f"Primary finding: {diagnosis_data.get('primary_finding', 'None')}. Differential: {differential_str}"
        
        return report

    def to_json(self, fhir_resource):
        return fhir_resource.json(indent=2)

if __name__ == "__main__":
    formatter = FHIRFormatter()
    sample_data = {
        "patient_id": "P12345",
        "primary_finding": "Silicosis",
        "differential": {"Silicosis": 0.72, "Pneumonia": 0.18, "Tuberculosis": 0.10}
    }
    report = formatter.create_diagnostic_report(sample_data)
    print(formatter.to_json(report))
