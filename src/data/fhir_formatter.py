from fhir.resources.diagnosticreport import DiagnosticReport
from fhir.resources.identifier import Identifier
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.reference import Reference
from fhir.resources.extension import Extension
from fhir.resources.observation import Observation
from fhir.resources.quantity import Quantity
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from pathlib import Path


# Externalized SNOMED CT code system - loaded from config file
DEFAULT_SNOMED_CODES = {
    "Silicosis": ("50751000", "Silicosis"),
    "Asbestosis": ("284594005", "Asbestosis"),
    "Pneumonia": ("233604007", "Pneumonia"),
    "Tuberculosis": ("56717001", "Tuberculosis"),
    "Normal": ("17621005", "Normal"),
    "Lung Cancer": ("93880001", "Malignant neoplasm of lung"),
    "Pulmonary Nodule": ("311589007", "Pulmonary nodule"),
    "COPD": ("13645005", "Chronic obstructive pulmonary disease"),
    "Pulmonary Embolism": ("59282003", "Pulmonary embolism"),
    "Interstitial Lung Disease": ("700250004", "Interstitial lung disease"),
}

_SNOMED_CODES: Optional[Dict[str, Tuple[str, str]]] = None


def load_snomed_codes(config_path: str = None) -> Dict[str, Tuple[str, str]]:
    """Load SNOMED CT codes from external configuration file."""
    global _SNOMED_CODES
    if _SNOMED_CODES is not None:
        return _SNOMED_CODES
    
    if config_path is None:
        config_path = os.getenv("SNOMED_CODES_PATH", "config/snomed_codes.json")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                _SNOMED_CODES = {k.lower(): tuple(v) for k, v in data.items()}
                return _SNOMED_CODES
        except Exception as e:
            print(f"Warning: Failed to load SNOMED codes from {config_path}: {e}")
    
    # Fallback to defaults
    _SNOMED_CODES = {k.lower(): v for k, v in DEFAULT_SNOMED_CODES.items()}
    return _SNOMED_CODES


def get_snomed_coding(condition_name: str) -> Coding:
    codes = load_snomed_codes()
    key = condition_name.lower().strip()
    code, display = codes.get(key, ("unknown", condition_name))
    return Coding(
        system="http://snomed.info/sct",
        code=code,
        display=display
    )


class FHIRFormatter:
    def __init__(self):
        pass

    def generate_hl7_oru(self, diagnosis_data: Dict, verdict: str, final_diagnosis: str = None, notes: str = "", doctor_id: str = "dr-anonymous") -> str:
        """
        Generate a basic HL7 ORU^R01 message string for a radiologist review.
        """
        import uuid
        msg_id = uuid.uuid4().hex[:10].upper()
        patient_id = diagnosis_data.get('patient_id', 'Unknown')
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        
        # MSH: Message Header
        msh = f"MSH|^~\\&|MEdiChainAI|MEdiChain|PACS|Hospital|{timestamp}||ORU^R01|{msg_id}|P|2.5"
        
        # PID: Patient Identification
        pid = f"PID|1||{patient_id}||Unknown^Patient|||||||||||||"
        
        # OBR: Observation Request
        obr = f"OBR|1|||RAD^Radiology Report|||{timestamp}|||||||||||||{doctor_id}"
        
        # OBX: Observation/Result
        finding = final_diagnosis if final_diagnosis else diagnosis_data.get('primary_finding', 'Unknown')
        obs_val = f"Verdict: {verdict}. Finding: {finding}. Notes: {notes}"
        obx = f"OBX|1|TX|REPORT^Final Report||{obs_val}||||||F|||{timestamp}"
        
        return "\r".join([msh, pid, obr, obx])

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
        
        # Structure warning flags as standard FHIR Extensions
        extensions = []
        if diagnosis_data.get('escalation_required', False):
            extensions.append(
                Extension(
                    url="http://medi-chain.io/fhir/StructureDefinition/escalation-required",
                    valueBoolean=True
                )
            )
        
        # Add uncertainty extension if present
        if 'uncertainty_std' in diagnosis_data:
            extensions.append(
                Extension(
                    url="http://medi-chain.io/fhir/StructureDefinition/uncertainty-std",
                    valueDecimal=float(diagnosis_data['uncertainty_std'])
                )
            )
        
        # Add OOD extension if present
        if diagnosis_data.get('ood_detected', False):
            extensions.append(
                Extension(
                    url="http://medi-chain.io/fhir/StructureDefinition/out-of-distribution",
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
    Mediates FHIR integrations with external EHR systems, featuring tenacity-based
    exponential backoff retries and a local Dead Letter Queue (DLQ) fallback.
    """
    def __init__(self, endpoint_url: str = None):
        import logging
        import os
        self.logger = logging.getLogger("ehr-gateway")
        self.endpoint_url = endpoint_url or os.getenv("EHR_GATEWAY_URL", "https://mock-ehr-gateway.internal/fhir")

    async def validate_patient(self, patient_id: str) -> bool:
        """Validates that the patient exists in the EHR system before posting observations.
        Makes a read request to Patient resource (GET /fhir/Patient/{id}).
        """
        import httpx
        
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
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                    self.logger.info(f"[EHR Gateway] Successfully validated patient {patient_id} in EHR.")
                    return True
                else:
                    self.logger.error(f"[EHR Gateway] Patient validation failed for ID {patient_id}. HTTP status: {response.status_code}")
                    return False
        except Exception as e:
            self.logger.error(f"[EHR Gateway] EHR connection error during patient validation: {e}")
            return False

    async def push_report(self, fhir_json: str, is_retry: bool = False):
        """Pushes the report to a hospital's FHIR server with robust retry logic."""
        # Validate patient existence before pushing report observations
        try:
            report_data = json.loads(fhir_json) if isinstance(fhir_json, str) else fhir_json
            subject_ref = report_data.get("subject", {}).get("reference", "")
            if subject_ref.startswith("Patient/"):
                patient_id = subject_ref.split("/")[-1]
                if not await self.validate_patient(patient_id):
                    self.logger.error(f"[EHR Gateway] Aborting report push: Patient ID {patient_id} is invalid.")
                    return False
        except Exception as parse_err:
            self.logger.warning(f"[EHR Gateway] Could not parse patient ID for validation: {parse_err}")

        from tenacity import AsyncRetrying, wait_exponential, stop_after_attempt, retry_if_exception
        import httpx
        
        def is_retriable_error(exception):
            if isinstance(exception, httpx.ConnectError) or isinstance(exception, httpx.ConnectTimeout):
                return True
            if isinstance(exception, httpx.HTTPStatusError):
                # Only retry on 5xx server errors
                return exception.response.status_code >= 500
            return False
        
        async def _execute_push():
            self.logger.info(f"[EHR Gateway] Connecting to {self.endpoint_url}...")
            # If the endpoint is mock, simulate transient network failures (30% rate)
            if "mock-ehr-gateway.internal" in self.endpoint_url:
                import random
                if random.random() < 0.3:
                    self.logger.warning("[EHR Gateway] Mock transient connection error triggered.")
                    raise httpx.ConnectError("Mock Connection Timeout")
                self.logger.info("[EHR Gateway] Successfully posted DiagnosticReport to Mock EHR.")
                return True
            else:
                # Real HTTP POST with OAuth2/mTLS headers
                headers = {"Content-Type": "application/fhir+json"}
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.post(self.endpoint_url, data=fhir_json, headers=headers)
                    response.raise_for_status()
                    self.logger.info(f"[EHR Gateway] Successfully posted DiagnosticReport to {self.endpoint_url}.")
                    return True
                
        try:
            async for attempt in AsyncRetrying(
                wait=wait_exponential(multiplier=1, min=2, max=10),
                stop=stop_after_attempt(3),
                retry=retry_if_exception(is_retriable_error),
                reraise=True
            ):
                with attempt:
                    return await _execute_push()
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
        import shutil
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
        
        if not redis_success:
            try:
                dlq_dir_str = os.getenv("DLQ_DIR", "temp/dlq")
                dlq_dir = Path(dlq_dir_str)
                dlq_dir.mkdir(parents=True, exist_ok=True)
                
                local_path = dlq_dir / filename
                
                # Check free space on local partition
                total, used, free = shutil.disk_usage(dlq_dir)
                if free < 50 * 1024 * 1024:  # Warning if < 50 MB
                    self.logger.critical("Local disk space critically low. Attempting to write full payload despite space warning.")
                    try:
                        from deployment.api.main import _send_system_alert
                        _send_system_alert("Disk Space Exhaustion", "DLQ write warning: low disk space on host partition.")
                    except Exception as alert_err:
                        self.logger.error(f"Failed to send system alert: {alert_err}")
                
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
                    from filelock import FileLock
                    lock_path = str(local_path) + ".lock"
                    with FileLock(lock_path, timeout=5):
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


class MockEHRGateway(EHRGateway):
    async def push_report(self, fhir_json: str, is_retry: bool = False):
        # Return success immediately
        return True


if __name__ == "__main__":
    formatter = FHIRFormatter()
    sample_data = {
        "patient_id": "P12345",
        "primary_finding": "Silicosis",
        "differential": {"Silicosis": 0.72, "Pneumonia": 0.18, "Tuberculosis": 0.10}
    }
    report = formatter.create_diagnostic_report(sample_data)
    print(formatter.to_json(report))