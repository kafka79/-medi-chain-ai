import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger("feedback-manager")

class FeedbackManager:
    """
    Manages clinician feedback to move from a 'Black Hole' to an 'Active Learning' pipeline.
    Aggregates feedback for periodic retraining and audit.
    """
    def __init__(self, feedback_dir: str = "data/feedback"):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.summary_file = self.feedback_dir / "summary.json"

    def log_feedback(self, case_id: str, diagnosis: str, clinician_correction: str, agreement: bool, comments: str):
        """Logs individual feedback entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "case_id": case_id,
            "system_diagnosis": diagnosis,
            "clinician_correction": clinician_correction,
            "agreement": agreement,
            "comments": comments
        }
        
        file_path = self.feedback_dir / f"feedback_{case_id}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(file_path, "w") as f:
            json.dump(entry, f, indent=4)
            
        self._update_summary(agreement)
        logger.info(f"Feedback logged for case {case_id}. Agreement: {agreement}")

    def _update_summary(self, agreement: bool):
        """Updates an aggregate summary for quick metrics review (addresses Jess's 'Metric Vacuum')."""
        summary = {"total_cases": 0, "agreements": 0, "disagreements": 0, "agreement_rate": 0.0}
        
        if self.summary_file.exists():
            with open(self.summary_file, "r") as f:
                summary = json.load(f)
        
        summary["total_cases"] += 1
        if agreement:
            summary["agreements"] += 1
        else:
            summary["disagreements"] += 1
            
        summary["agreement_rate"] = summary["agreements"] / summary["total_cases"]
        
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=4)

    def get_discrepancy_report(self) -> List[Dict[str, Any]]:
        """Returns all cases where the clinician disagreed for priority review."""
        discrepancies = []
        for file in self.feedback_dir.glob("feedback_*.json"):
            with open(file, "r") as f:
                entry = json.load(f)
                if not entry.get("agreement", True):
                    discrepancies.append(entry)
        return discrepancies
