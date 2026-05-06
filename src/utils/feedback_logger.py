import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import uuid


class FeedbackLogger:
    def __init__(self, output_dir: str = "outputs/feedback"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "physician_feedback.csv"

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

        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(record)

        return self.csv_path
