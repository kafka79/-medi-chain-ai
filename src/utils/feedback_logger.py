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

        # 3. Fallback to local append
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(record)

        return self.csv_path
