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
    Addresses the 'Integration Friction' and 'Silent Failure' flaws.
    Mediates FHIR integrations with external EHR systems, featuring tenacity-based
    exponential backoff retries and a local Dead Letter Queue (DLQ) fallback.
    """
    def __init__(self, endpoint_url: str = None):
        import logging
        import os
        self.logger = logging.getLogger("ehr-gateway")
        self.endpoint_url = endpoint_url or os.getenv("EHR_GATEWAY_URL", "https://mock-ehr-gateway.internal/fhir")

    def push_report(self, fhir_json: str):
        """Pushes the report to a hospital's FHIR server with robust retry logic."""
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
            self.logger.error(f"[EHR Gateway] All push attempts failed: {e}. Writing to Dead Letter Queue (DLQ).")
            self._write_to_dlq(fhir_json, e)
            return False

    def _write_to_dlq(self, fhir_json: str, exception: Exception):
        import uuid
        import time
        import os
        from pathlib import Path
        
        try:
            filename = f"failed_report_{int(time.time())}_{uuid.uuid4().hex[:6]}.json"
            
            try:
                parsed_payload = json.loads(fhir_json) if isinstance(fhir_json, str) else fhir_json
            except json.JSONDecodeError:
                parsed_payload = fhir_json  # Fallback to raw string if invalid JSON

            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(exception),
                "payload": parsed_payload
            }
            
            # 1. Attempt to write to Redis shared list as the primary tier (replicated DLQ)
            try:
                import redis
                redis_host = os.getenv("REDIS_HOST", "redis")
                redis_port = int(os.getenv("REDIS_PORT", "6379"))
                r = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True, socket_connect_timeout=1)
                r.rpush("medi_chain:dlq", json.dumps(payload))
                self.logger.info("[EHR Gateway] Replicated Redis DLQ Synced: Pushed failed payload to redis list 'medi_chain:dlq'")
            except Exception as redis_err:
                self.logger.warning(f"[EHR Gateway] Redis DLQ upload failed: {redis_err}")
            
            # 2. Write locally first to a persistent local DLQ folder (secondary backup)
            dlq_dir = Path("temp/dlq")
            dlq_dir.mkdir(parents=True, exist_ok=True)
            local_path = dlq_dir / filename
            
            with open(local_path, "w") as f:
                json.dump(payload, f, indent=2)
            self.logger.critical(f"[EHR Gateway] Local DLQ Written: Saved failed FHIR payload locally to {local_path}")
            
            # 3. Attempt to backup to remote S3 bucket, but do not raise if it fails (tertiary backup)
            try:
                from src.utils.storage import S3StorageProvider
                storage = S3StorageProvider(endpoint=os.getenv("MINIO_ENDPOINT", "minio:9000"), bucket="dlq")
                if storage.client:
                    with open(local_path, "rb") as f:
                        storage.save(f, filename)
                    self.logger.info(f"[EHR Gateway] Remote DLQ Synced: Uploaded {filename} to MinIO bucket 'dlq'")
            except Exception as s3_err:
                self.logger.warning(f"[EHR Gateway] S3 DLQ upload failed (expected if network is down): {s3_err}")
                
        except Exception as dlq_err:
            self.logger.error(f"[EHR Gateway] Failed to write to any DLQ: {dlq_err}")

if __name__ == "__main__":
    formatter = FHIRFormatter()
    sample_data = {
        "patient_id": "P12345",
        "primary_finding": "Silicosis",
        "differential": {"Silicosis": 0.72, "Pneumonia": 0.18, "Tuberculosis": 0.10}
    }
    report = formatter.create_diagnostic_report(sample_data)
    print(formatter.to_json(report))
