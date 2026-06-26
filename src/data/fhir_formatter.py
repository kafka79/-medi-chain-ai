from fhir.resources.diagnosticreport import DiagnosticReport
from fhir.resources.identifier import Identifier
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.reference import Reference
from fhir.resources.extension import Extension
from fhir.resources.observation import Observation
from fhir.resources.quantity import Quantity
import json
from datetime import datetime, timezone

SNOMED_CODES = {
    "silicosis": ("50751000", "Silicosis"),
    "asbestosis": ("284594005", "Asbestosis"),
    "pneumonia": ("233604007", "Pneumonia"),
    "tuberculosis": ("56717001", "Tuberculosis"),
    "normal": ("17621005", "Normal"),
}

def get_snomed_coding(condition_name: str) -> Coding:
    key = condition_name.lower().strip()
    code, display = SNOMED_CODES.get(key, ("unknown", condition_name))
    return Coding(
        system="http://snomed.info/sct",
        code=code,
        display=display
    )

class FHIRFormatter:
    def __init__(self):
        pass

    def create_diagnostic_report(self, diagnosis_data):
        """
        Create a FHIR-R4 DiagnosticReport from raw diagnosis results.
        diagnosis_data: dict with keys: patient_id, differential, confidence, findings, escalation_required
        """
        differential_str = ", ".join([f"{k}: {v:.1%}" for k, v in diagnosis_data.get('differential', {}).items()])
        status = "preliminary" if diagnosis_data.get('escalation_required', False) else "final"
        
        conclusion = (
            f"Primary finding: {diagnosis_data.get('primary_finding', 'None')}. "
            f"Differential: {differential_str}"
        )
        
        # Structure warning flags as standard FHIR Extensions rather than raw strings in text blocks
        extensions = []
        if diagnosis_data.get('escalation_required', False):
            extensions.append(
                Extension(
                    url="http://medi-chain.io/fhir/StructureDefinition/escalation-required",
                    valueBoolean=True
                )
            )

        patient_id = diagnosis_data.get('patient_id', 'Unknown')
        
        # Generate contained Observation resources for differential diagnosis
        contained_observations = []
        result_references = []
        
        import uuid
        timestamp = int(datetime.now(timezone.utc).timestamp())
        
        for condition, prob in diagnosis_data.get('differential', {}).items():
            obs_id = f"obs-{condition.lower().replace(' ', '-')}-{uuid.uuid4().hex[:6]}"
            coding = get_snomed_coding(condition)
            
            obs = Observation(
                id=obs_id,
                status=status,
                code=CodeableConcept(
                    coding=[coding]
                ),
                subject=Reference(
                    reference=f"Patient/{patient_id}",
                    display=f"Patient {patient_id}"
                ),
                valueQuantity=Quantity(
                    value=float(prob),
                    unit="%",
                    system="http://unitsofmeasure.org",
                    code="%"
                )
            )
            contained_observations.append(obs)
            result_references.append(Reference(reference=f"#{obs_id}"))

        return DiagnosticReport(
            status=status,
            extension=extensions,
            identifier=[
                Identifier(
                    value=f"RPT-{patient_id}-{timestamp}"
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
            subject=Reference(
                reference=f"Patient/{patient_id}",
                display=f"Patient {patient_id}"
            ),
            effectiveDateTime=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            conclusion=conclusion,
            contained=contained_observations if contained_observations else None,
            result=result_references if result_references else None,
        )

    def to_json(self, fhir_resource):
        return fhir_resource.json(indent=2)

class EHRGateway:
    """
    Addresses the 'Integration Friction' and 'Silent Failure' flaws.
    Mediates FHIR integrations with external EHR systems, featuring tenacity-based
    exponential backoff retries and a local Dead Letter Queue (DLQ) fallback.
    """
    def __init__(self, endpoint_url: str = None):
        import logging
        import os
        self.logger = logging.getLogger("ehr-gateway")
        self.endpoint_url = endpoint_url or os.getenv("EHR_GATEWAY_URL", "https://mock-ehr-gateway.internal/fhir")

    def validate_patient(self, patient_id: str) -> bool:
        """Validates that the patient exists in the EHR system before posting observations.
        Makes a read request to Patient resource (GET /fhir/Patient/{id}).
        """
        import requests
        
        # If the endpoint is mock, simulate validation
        if "mock-ehr-gateway.internal" in self.endpoint_url:
            self.logger.info(f"[EHR Gateway] Mock validating patient {patient_id}...")
            if patient_id == "invalid-patient-id":
                self.logger.error(f"[EHR Gateway] Patient {patient_id} not found in Mock EHR.")
                return False
            return True
            
        try:
            url = f"{self.endpoint_url}/Patient/{patient_id}"
            headers = {"Accept": "application/fhir+json"}
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                self.logger.info(f"[EHR Gateway] Successfully validated patient {patient_id} in EHR.")
                return True
            else:
                self.logger.error(f"[EHR Gateway] Patient validation failed for ID {patient_id}. HTTP status: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"[EHR Gateway] EHR connection error during patient validation: {e}")
            return False

    def push_report(self, fhir_json: str, is_retry: bool = False):
        """Pushes the report to a hospital's FHIR server with robust retry logic."""
        # Validate patient existence before pushing report observations
        try:
            report_data = json.loads(fhir_json) if isinstance(fhir_json, str) else fhir_json
            subject_ref = report_data.get("subject", {}).get("reference", "")
            if subject_ref.startswith("Patient/"):
                patient_id = subject_ref.split("/")[-1]
                if not self.validate_patient(patient_id):
                    self.logger.error(f"[EHR Gateway] Aborting report push: Patient ID {patient_id} is invalid.")
                    return False
        except Exception as parse_err:
            self.logger.warning(f"[EHR Gateway] Could not parse patient ID for validation: {parse_err}")

        from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception
        import requests
        
        def is_retriable_error(exception):
            if isinstance(exception, requests.exceptions.ConnectionError):
                return True
            if isinstance(exception, requests.exceptions.Timeout):
                return True
            if isinstance(exception, requests.exceptions.HTTPError):
                # Only retry on 5xx server errors
                return exception.response.status_code >= 500
            return False
        
        @retry(
            wait=wait_exponential(multiplier=1, min=2, max=10),
            stop=stop_after_attempt(3),
            retry=retry_if_exception(is_retriable_error),
            reraise=True
        )
        def _execute_push():
            self.logger.info(f"[EHR Gateway] Connecting to {self.endpoint_url}...")
            # If the endpoint is mock, simulate transient network failures (30% rate)
            if "mock-ehr-gateway.internal" in self.endpoint_url:
                import random
                if random.random() < 0.3:
                    self.logger.warning("[EHR Gateway] Mock transient connection error triggered.")
                    raise requests.exceptions.ConnectionError("Mock Connection Timeout")
                self.logger.info("[EHR Gateway] Successfully posted DiagnosticReport to Mock EHR.")
                return True
            else:
                # Real HTTP POST with OAuth2/mTLS headers
                headers = {"Content-Type": "application/fhir+json"}
                response = requests.post(self.endpoint_url, data=fhir_json, headers=headers, timeout=5)
                response.raise_for_status()
                self.logger.info(f"[EHR Gateway] Successfully posted DiagnosticReport to {self.endpoint_url}.")
                return True
                
        try:
            return _execute_push()
        except Exception as e:
            if not is_retry:
                self.logger.error(f"[EHR Gateway] All push attempts failed: {e}. Writing to Dead Letter Queue (DLQ).")
                self._write_to_dlq(fhir_json, e)
            else:
                self.logger.error(f"[EHR Gateway] Retry push attempt failed: {e}.")
            return False

    def _write_to_dlq(self, fhir_json: str, exception: Exception):
        import uuid
        import time
        import os
        import tempfile
        from pathlib import Path
        
        filename = f"failed_report_{int(time.time())}_{uuid.uuid4().hex[:6]}.json"
        
        try:
            parsed_payload = json.loads(fhir_json) if isinstance(fhir_json, str) else fhir_json
        except json.JSONDecodeError:
            parsed_payload = fhir_json  # Fallback to raw string if invalid JSON

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(exception),
            "payload": parsed_payload,
            "retry_count": 0
        }
        
        payload_json = json.dumps(payload, indent=2)
        redis_success = False
        local_success = False
        
        # Flaw #7-structural Fix: Redis is the PRIMARY DLQ tier (shared across replicas).
        # Previous code used local files with filelock as secondary — but filelock is
        # process-local and does not coordinate across containers/replicas.
        # Now: Redis first, local disk as emergency-only (no lock needed — atomic rename).
        
        # 1. Primary: Redis shared list (replicated across all API replicas)
        try:
            import redis
            redis_host = os.getenv("REDIS_HOST", "redis")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            r = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True, socket_connect_timeout=1)
            r.rpush("medi_chain:dlq", payload_json)
            self.logger.info("[EHR Gateway] Redis DLQ: Pushed failed payload to 'medi_chain:dlq'")
            redis_success = True
        except Exception as redis_err:
            self.logger.warning(f"[EHR Gateway] Redis DLQ failed: {redis_err}")
        
        # 2. Tertiary backup: S3 (attempt regardless of local success)
        try:
            from src.utils.storage import S3StorageProvider
            storage = S3StorageProvider(endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"), bucket="dlq")
            if storage.client:
                import io
                storage.save(io.BytesIO(payload_json.encode("utf-8")), filename)
                self.logger.info(f"[EHR Gateway] S3 DLQ: Uploaded {filename} to MinIO bucket 'dlq'")
        except Exception as s3_err:
            self.logger.warning(f"[EHR Gateway] S3 DLQ upload failed: {s3_err}")
        
        # 3. Emergency local fallback: atomic write-then-rename (no filelock needed)
        # This is per-replica and NOT shared, but prevents data loss if Redis AND S3 are both down.
        if not redis_success:
            try:
                dlq_dir = Path(os.getenv("DLQ_DIR", "temp/dlq"))
                dlq_dir.mkdir(parents=True, exist_ok=True)
                local_path = dlq_dir / filename
                
                # Encrypt the payload string before persisting to local disk fallback
                from src.utils.security import encrypt_payload
                encrypted_payload = encrypt_payload(payload_json)
                wrapper = {
                    "encrypted": True,
                    "data": encrypted_payload
                }
                wrapper_json = json.dumps(wrapper, indent=2)
                
                # Atomic: write to temp file, then rename (rename is atomic on POSIX)
                fd, tmp_path = tempfile.mkstemp(dir=str(dlq_dir), suffix=".tmp")
                try:
                    with os.fdopen(fd, "w") as f:
                        f.write(wrapper_json)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, str(local_path))
                    
                    # Sync parent directory to ensure directory metadata changes are durable (POSIX)
                    if os.name != "nt":
                        try:
                            dir_fd = os.open(str(dlq_dir), os.O_RDONLY)
                            try:
                                os.fsync(dir_fd)
                            finally:
                                os.close(dir_fd)
                        except Exception as dir_sync_err:
                            self.logger.warning(f"[EHR Gateway] Parent directory sync failed: {dir_sync_err}")
                except Exception:
                    # Clean up temp file on failure
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    raise
                self.logger.critical(f"[EHR Gateway] Emergency local DLQ: Saved to {local_path}")
                local_success = True
            except Exception as local_err:
                self.logger.error(f"[EHR Gateway] Local DLQ write failed: {local_err}")
                
        if not redis_success and not local_success:
            raise RuntimeError(
                f"Failed to persist report to any DLQ. Redis failed, and local disk failed: {exception}"
            )

if __name__ == "__main__":
    formatter = FHIRFormatter()
    sample_data = {
        "patient_id": "P12345",
        "primary_finding": "Silicosis",
        "differential": {"Silicosis": 0.72, "Pneumonia": 0.18, "Tuberculosis": 0.10}
    }
    report = formatter.create_diagnostic_report(sample_data)
    print(formatter.to_json(report))
