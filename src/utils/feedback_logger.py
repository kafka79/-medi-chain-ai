import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import uuid
import io
import json
import logging

logger = logging.getLogger("feedback-logger")


class FeedbackLogger:
    def __init__(self, output_dir: str = "outputs/feedback", redis_client: Optional[Any] = None, storage_provider: Optional[Any] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "physician_feedback.csv"
        self.redis_client = redis_client
        self.storage_provider = storage_provider

    def log_feedback(
        self,
        *,
        session_id: str,
        verdict: str,
        notes: str,
        diagnosis: Dict[str, Any],
        history_metadata: Dict[str, Any],
        disagreement_reason: Optional[str] = None,
        correction_mask_base64: Optional[str] = None,
    ) -> Path:
        record = {
            "feedback_id": uuid.uuid4().hex,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "verdict": verdict,
            "notes": notes.strip(),
            "top_finding": diagnosis.get("top_finding", "Unknown"),
            "uncertainty_std": diagnosis.get("uncertainty_std", ""),
            "patient_id": history_metadata.get("patient_id", "Unknown"),
            "occupation": history_metadata.get("occupation", "Unknown"),
            "disagreement_reason": disagreement_reason or "",
            "correction_mask": correction_mask_base64 or "",
        }

        # 1. Distributed sync to Redis if available
        if self.redis_client is not None:
            try:
                self.redis_client.rpush("medi_chain:feedback:records", json.dumps(record))
                logger.info(f"Feedback {record['feedback_id']} pushed to shared Redis list.")
            except Exception as e:
                logger.error(f"Failed to push feedback to Redis list: {e}")

        # 2. Distributed sync to S3 if available
        if self.storage_provider is not None:
            try:
                feedback_json = json.dumps(record, indent=4)
                file_obj = io.BytesIO(feedback_json.encode("utf-8"))
                self.storage_provider.save(file_obj, f"feedback/feedback_{record['feedback_id']}.json")
                logger.info(f"Feedback {record['feedback_id']} saved to S3 bucket.")
            except Exception as e:
                logger.error(f"Failed to save feedback to S3: {e}")

        # 3. Fallback to local append (configurable, purely instance-local without POSIX locks)
        import os
        import tempfile
        if os.getenv("ENABLE_LOCAL_CSV_LOGGING", "true").lower() == "true":
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                write_header = not self.csv_path.exists()
                
                # Atomic write: Read existing, append to temp file, then rename
                fd, tmp_path = tempfile.mkstemp(dir=str(self.output_dir), suffix=".tmp")
                try:
                    with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
                        
                        # Copy existing content if it exists
                        if self.csv_path.exists():
                            with open(self.csv_path, "r", encoding="utf-8") as old_handle:
                                handle.write(old_handle.read())
                        elif write_header:
                            writer.writeheader()
                            
                        writer.writerow(record)
                        handle.flush()
                        os.fsync(handle.fileno())
                    
                    # Atomic replace
                    os.replace(tmp_path, str(self.csv_path))
                except Exception:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    raise
            except Exception as e:
                logger.error(f"Failed to append local feedback CSV: {e}")

        return self.csv_path
