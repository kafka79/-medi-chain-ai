import csv
import json
from pathlib import Path
import shutil
import uuid

from src.evaluation.report_generator import ClinicalReportGenerator
from src.utils.feedback_logger import FeedbackLogger


def make_temp_dir() -> Path:
    base = Path("outputs/reports") / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    return base


def test_feedback_logger_persists_records():
    base_dir = make_temp_dir()
    try:
        logger = FeedbackLogger(output_dir=str(base_dir))
        csv_path = logger.log_feedback(
            session_id="session-1",
            verdict="Match",
            notes="Looks correct",
            diagnosis={"top_finding": "Normal", "uncertainty_std": 0.03},
            history_metadata={"patient_id": "P100", "occupation": "Welder"},
        )

        assert csv_path.exists()
        with csv_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        assert len(rows) == 1
        assert rows[0]["session_id"] == "session-1"
        assert rows[0]["verdict"] == "Match"
        assert rows[0]["patient_id"] == "P100"
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)


def test_report_generator_writes_pdf_and_fhir():
    base_dir = make_temp_dir()
    try:
        generator = ClinicalReportGenerator(output_dir=str(base_dir))
        report_bundle = generator.generate_report(
            diagnosis_result={
                "top_finding": "Silicosis",
                "confidence": 0.72,
                "uncertainty_std": 0.08,
                "probabilities": [0.72, 0.18, 0.05, 0.03, 0.02],
                "escalation_required": False,
            },
            patient_metadata={
                "patient_id": "P999",
                "age": 54,
                "gender": "Male",
                "occupation": "Driller",
            },
            heatmap_path=str(base_dir / "missing_heatmap.png"),
            citations=[{"title": "Example Study", "pmid": "12345", "text": "Abstract text"}],
            output_filename="report.pdf",
        )

        assert report_bundle["pdf_path"].endswith(".pdf")
        assert report_bundle["fhir_path"].endswith(".fhir.json")

        with open(report_bundle["fhir_path"], "r", encoding="utf-8") as handle:
            fhir_payload = json.load(handle)

        assert fhir_payload["resourceType"] == "DiagnosticReport"
        assert "Silicosis" in fhir_payload["conclusion"]
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
