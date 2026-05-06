from fhir.resources.diagnosticreport import DiagnosticReport
from fhir.resources.identifier import Identifier
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.reference import Reference
import json
from datetime import datetime, timezone

class FHIRFormatter:
    def __init__(self):
        pass

    def create_diagnostic_report(self, diagnosis_data):
        """
        Create a FHIR-R4 DiagnosticReport from raw diagnosis results.
        diagnosis_data: dict with keys: patient_id, differential, confidence, findings
        """
        differential_str = ", ".join([f"{k}: {v:.1%}" for k, v in diagnosis_data.get('differential', {}).items()])
        return DiagnosticReport(
            status="final",
            identifier=[
                Identifier(
                    value=f"RPT-{diagnosis_data.get('patient_id', 'UNK')}-{int(datetime.now().timestamp())}"
                )
            ],
            category=[
                CodeableConcept(
                    coding=[
                        Coding(
                            system="http://terminology.hl7.org/CodeSystem/v2-0074",
                            code="RAD",
                            display="Radiology",
                        )
                    ]
                )
            ],
            code=CodeableConcept(
                coding=[
                    Coding(
                        system="http://loinc.org",
                        code="18748-4",
                        display="Diagnostic imaging report",
                    )
                ]
            ),
            subject=Reference(display=f"Patient {diagnosis_data.get('patient_id', 'Unknown')}"),
            effectiveDateTime=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            conclusion=(
                f"Primary finding: {diagnosis_data.get('primary_finding', 'None')}. "
                f"Differential: {differential_str}"
            ),
        )

    def to_json(self, fhir_resource):
        return fhir_resource.json(indent=2)

class EHRGateway:
    """
    Addresses the 'Integration Friction' flaw.
    Acts as a mediator between MEdi Chain AI and external EHR systems (Epic/Cerner/OpenEMR).
    """
    def __init__(self, endpoint_url: str = None):
        self.endpoint_url = endpoint_url or "https://mock-ehr-gateway.internal/fhir"

    def push_report(self, fhir_json: str):
        """Simulates pushing the report to a hospital's FHIR server."""
        print(f"[EHR Gateway] Connecting to {self.endpoint_url}...")
        # In a real scenario, this would use OAuth2/M mTLS and POST to the FHIR endpoint
        try:
            # Simulation of a successful POST request
            print(f"[EHR Gateway] Successfully posted DiagnosticReport to EHR.")
            return True
        except Exception as e:
            print(f"[EHR Gateway] Failed to integrate with EHR: {e}")
            return False

if __name__ == "__main__":
    formatter = FHIRFormatter()
    sample_data = {
        "patient_id": "P12345",
        "primary_finding": "Silicosis",
        "differential": {"Silicosis": 0.72, "Pneumonia": 0.18, "Tuberculosis": 0.10}
    }
    report = formatter.create_diagnostic_report(sample_data)
    print(formatter.to_json(report))
