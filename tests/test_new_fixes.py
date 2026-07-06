import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import asyncio

@pytest.mark.asyncio
async def test_worker_cli_arguments():
    """Verify worker.py correctly parses command-line arguments to run process-isolated tasks."""
    import sys
    
    # Save original sys.argv
    orig_argv = sys.argv
    try:
        # Run cleanup standalone task CLI option mock
        sys.argv = ["worker.py", "--task", "cleanup"]
        with patch("src.worker.run_cleanup") as mock_cleanup, \
             patch("src.worker.run_dlq_reconciliation") as mock_dlq, \
             patch("src.worker.threading.Thread") as mock_thread:
            
            from src.worker import main
            main()
            mock_cleanup.assert_called_once()
            mock_dlq.assert_not_called()
            mock_thread.assert_not_called()
            
        # Run dlq standalone task CLI option mock
        sys.argv = ["worker.py", "--task", "dlq"]
        with patch("src.worker.run_cleanup") as mock_cleanup, \
             patch("src.worker.run_dlq_reconciliation") as mock_dlq, \
             patch("src.worker.threading.Thread") as mock_thread:
            
            main()
            mock_cleanup.assert_not_called()
            mock_dlq.assert_called_once()
            mock_thread.assert_not_called()
            
    finally:
        sys.argv = orig_argv

@pytest.mark.asyncio
async def test_langgraph_schema_migrations():
    """Verify ClinicalAgent automatically performs state schema migrations at node execution."""
    from src.agent.clinical_graph import ClinicalAgent
    
    # Mock dependencies
    mock_parser = MagicMock()
    mock_rag = MagicMock()
    
    with patch.dict("os.environ", {"INTERNAL_API_KEY": "dummy"}):
        agent = ClinicalAgent(mock_parser, mock_rag, inference_api_url="http://dummy")
        
        # Build an older version 1 state dictionary (missing schema_version and heatmap_base64)
        legacy_state = {
            "image_path": "scan.jpg",
            "patient_pdf_path": "history.pdf",
            "iteration_count": 0,
            "escalation_required": False,
            "pubmed_citations": [],
            "visual_features": None,
            "visual_std": None,
            "history_data": {},
            "diagnosis": {},
            "confidence": 0.0
        }
        
        # Run schema migration manually to verify
        migrated = agent._ensure_current_schema(legacy_state)
        
        assert migrated["schema_version"] == 2
        assert "heatmap_base64" in migrated
        assert migrated["heatmap_base64"] == ""
        
        # Verify legacy state wasn't mutated in-place
        assert "schema_version" not in legacy_state

@pytest.mark.asyncio
async def test_dynamic_biomedical_concept_extraction():
    """Verify that ClinicalAgent dynamically extracts concepts and does not rely on static hardcoded values."""
    from src.agent.clinical_graph import ClinicalAgent
    
    mock_parser = MagicMock()
    mock_rag = MagicMock()
    
    with patch.dict("os.environ", {"INTERNAL_API_KEY": "dummy"}):
        agent = ClinicalAgent(mock_parser, mock_rag, inference_api_url="http://dummy")
        
        # Test Case 1: Silicosis quarry exposure
        concepts1 = agent._extract_biomedical_concepts(
            chief_complaint="Severe cough and dyspnea in quarry worker",
            pmh="History of sandblasting and suspected silicosis"
        )
        assert "silicosis" in concepts1
        assert "dyspnea" in concepts1
        
        # Test Case 2: Rare beryllium exposure
        concepts2 = agent._extract_biomedical_concepts(
            chief_complaint="Chest tightness, dyspnea, and dry cough",
            pmh="Beryllium exposure from aerospace manufacturing sector"
        )
        assert "berylliosis" in concepts2
        assert "tightness" in concepts2 or "aerospace" in concepts2

@pytest.mark.asyncio
async def test_cryptographic_audit_log_chaining():
    """Verify SecureAuditLogger restricts file permissions and implements hash-chained logging."""
    from src.data.privacy_scrubber import SecureAuditLogger
    
    temp_dir = tempfile.mkdtemp()
    log_file = Path(temp_dir) / "audit.log"
    
    try:
        logger = SecureAuditLogger(filepath=str(log_file))
        
        # Write record 1
        record1 = {"operation": "scrub", "pii_detected": True, "timestamp": "2026-07-06T12:00:00Z"}
        logger.log_record(record1)
        
        # Write record 2
        record2 = {"operation": "scrub", "pii_detected": False, "timestamp": "2026-07-06T12:01:00Z"}
        logger.log_record(record2)
        
        # Verify file exists and has content
        assert log_file.exists()
        
        # Check permissions (only owner read/write on support systems)
        if os.name != "nt":
            mode = os.stat(log_file).st_mode & 0o777
            assert mode == 0o600
            
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        assert len(lines) == 2
        r1 = json.loads(lines[0])
        r2 = json.loads(lines[1])
        
        # Verify cryptographic hash chaining
        assert r1["previous_hash"] == "0" * 64
        assert r2["previous_hash"] == r1["hash"]
        assert r1["hash"] != r2["hash"]
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
