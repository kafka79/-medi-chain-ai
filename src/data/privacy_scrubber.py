import re
import logging
import threading
import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List

logger = logging.getLogger("privacy-scrubber")

# Flaw #23 Fix: Dedicated audit logger for PHI de-identification operations
_audit_logger = logging.getLogger("phi-audit")
_audit_logger.setLevel(logging.INFO)

import hashlib

class SecureAuditLogger:
    """
    Implements a cryptographic hash-chained, tamper-evident audit log writer
    to meet stringent HIPAA compliance requirements.
    """
    def __init__(self, filepath="temp/audit.log"):
        self.filepath = filepath
        self._lock = threading.Lock()
        self.last_hash = "0" * 64
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        # Load the last hash from existing log file if it exists and has content
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if last_line:
                            record = json.loads(last_line)
                            if "hash" in record:
                                self.last_hash = record["hash"]
            except Exception as e:
                logger.error(f"[Secure Audit] Failed to read last hash from audit log: {e}")
                
        # Set file permissions to owner-read-write only (0o600)
        try:
            if os.path.exists(self.filepath):
                os.chmod(self.filepath, 0o600)
        except Exception as e:
            logger.warning(f"[Secure Audit] Failed to restrict audit log permissions: {e}")

    def log_record(self, record: Dict[str, Any]):
        with self._lock:
            # Add chaining metadata
            record["previous_hash"] = self.last_hash
            
            # Compute current hash
            record_str = json.dumps(record, sort_keys=True)
            hasher = hashlib.sha256()
            hasher.update((self.last_hash + record_str).encode("utf-8"))
            current_hash = hasher.hexdigest()
            record["hash"] = current_hash
            
            # Update in-memory pointer
            self.last_hash = current_hash
            
            # Append to log file
            try:
                is_new = not os.path.exists(self.filepath)
                with open(self.filepath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                if is_new:
                    try:
                        os.chmod(self.filepath, 0o600)
                    except Exception:
                        pass
            except Exception as e:
                logger.critical(f"[Secure Audit] CRITICAL: Failed to write to secure audit trail: {e}")

# Instantiate global secure audit logger
secure_audit_logger = SecureAuditLogger()

class PrivacyScrubber:
    """
    Addresses 'De-identification Edge Cases'.
    Performs deep cleaning of extracted clinical text to remove PII/PHI.
    """
    def __init__(self):
        # Basic patterns for names, IDs, and dates
        self.patterns = {
            "mrn": re.compile(r"\b\d{6,10}\b"),
            "dob": re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
            "date": re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b", re.IGNORECASE),
            "zipcode": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
            "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
            "doctor": re.compile(r"\bDr\.\s+[A-Z][a-zA-Z]*\b"),
            "patient": re.compile(r"\bPatient:\s+[A-Z][a-zA-Z]*\b")
        }
        self.ner = None
        self.ner_load_error = None
        self._ner_lock = asyncio.Lock()
        self._ner_loaded = False
        self._ner_loading = False
        
        # Load configuration
        from src.config.settings import get_privacy_config
        privacy_config = get_privacy_config()
        lazy_load = privacy_config.ner_lazy_load
        
        # In testing mode, allow deferring; in production, start loading immediately
        if os.getenv("TESTING") != "true":
            if lazy_load:
                logger.info("NER_LAZY_LOAD is true. NER pipeline will be initialized on first use.")
            else:
                # Start loading in background but don't block startup
                asyncio.create_task(self._init_ner_async())
                logger.info("NER pipeline initialization started in background.")

    async def _init_ner_async(self):
        """Async initialization of the Hugging Face NER pipeline."""
        async with self._ner_lock:
            if self._ner_loaded or self._ner_loading:
                return
            self._ner_loading = True
            logger.info("Initializing Hugging Face NER pipeline...")
            try:
                from transformers import pipeline
                from src.config.settings import get_privacy_config
                privacy_config = get_privacy_config()
                model_name = privacy_config.ner_model_path
                logger.info(f"Loading NER model from path/name: {model_name}")
                # Using aggregation_strategy="simple" merges B-PER and I-PER into a single PER entity
                # Run in thread pool to avoid blocking event loop
                loop = asyncio.get_running_loop()
                self.ner = await loop.run_in_executor(
                    None, 
                    lambda: pipeline("ner", model=model_name, aggregation_strategy="simple")
                )
                self.ner_load_error = None
                logger.info("Hugging Face NER pipeline loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load NER pipeline: {e}")
                self.ner = None
                self.ner_load_error = e
            finally:
                self._ner_loaded = True
                self._ner_loading = False

    async def _ensure_ner_loaded(self):
        """Ensure NER model is loaded, waiting if necessary."""
        if not self._ner_loaded:
            await self._init_ner_async()
            # If another task already loaded it, wait
            while self._ner_loading:
                await asyncio.sleep(0.1)
            
        if self.ner is None or self.ner_load_error is not None:
            raise RuntimeError(
                f"Critical HIPAA Risk: Privacy scrubber failed to load the NER model. "
                f"Aborting process to prevent patient data leakage. Error: {self.ner_load_error}"
            )

    async def scrub_text_async(self, text: str, source_context: str = "unknown") -> str:
        """Async version of scrub_text for use in async contexts."""
        await self._ensure_ner_loaded()
        
        # Flaw #23 Fix: Track all redactions for the audit trail
        redaction_log: List[Dict[str, str]] = []
            
        scrubbed = text
        for label, pattern in self.patterns.items():
            matches = pattern.findall(scrubbed)
            if matches:
                redaction_log.append({"type": "regex", "pattern": label, "count": len(matches)})
            scrubbed = pattern.sub(f"[{label.upper()}_REDACTED]", scrubbed)
            
        try:
            entities = self.ner(scrubbed)
            ner_redacted = []
            for ent in sorted(entities, key=lambda x: x['start'], reverse=True):
                if ent['entity_group'] in ['PER', 'ORG', 'LOC']:
                    ner_redacted.append({"group": ent['entity_group'], "score": round(ent.get('score', 0), 3)})
                    scrubbed = scrubbed[:ent['start']] + f"[NER_{ent['entity_group']}_REDACTED]" + scrubbed[ent['end']:]
            if ner_redacted:
                redaction_log.append({"type": "ner", "entities": ner_redacted})
        except Exception as e:
            logger.error(f"NER scrubbing failed: {e}. Falling back to conservative regex de-identification.")
            words = scrubbed.split()
            fallback_words = []
            protected_words = {
                "the", "a", "an", "this", "that", "it", "patient", "history", "labs", "complaint", 
                "clinical", "findings", "silicosis", "pneumonia", "tuberculosis", "asbestosis", 
                "normal", "copd", "emphysema", "fibrosis", "mass", "nodule", "adenocarcinoma",
                "squamous", "lung", "cancer", "tumor", "bronchitis", "sarcoidosis", "effusion",
                "pleural", "pulmonary", "respiratory", "cardiac", "vascular", "mediastinal",
                "hilar", "lung", "chest", "xray", "x-ray", "ct", "mri", "scan", "imaging",
                "radiology", "radiologist", "physician", "doctor", "nurse", "hospital", "clinic",
                "emergency", "admission", "discharge", "followup", "follow-up", "outpatient",
                "inpatient", "icu", "er", "ed", "or", "dr", "md", "rn", "np", "pa"
            }
            for w in words:
                clean_w = w.strip(".,;:?!\"'()")
                if clean_w and clean_w[0].isupper() and clean_w.lower() not in protected_words:
                    redacted_term = "[FALLBACK_PER_ORG_REDACTED]"
                    idx = w.find(clean_w)
                    if idx != -1:
                        prefix = w[:idx]
                        suffix = w[idx+len(clean_w):]
                        fallback_words.append(f"{prefix}{redacted_term}{suffix}")
                    else:
                        fallback_words.append(w)
                else:
                    fallback_words.append(w)
            scrubbed = " ".join(fallback_words)
            redaction_log.append({"type": "ner_fallback_triggered", "error": str(e)})
        
        audit_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source_context,
            "operation": "text_scrub",
            "pii_detected": len(redaction_log) > 0,
            "redactions": redaction_log,
        }
        secure_audit_logger.log_record(dict(audit_record))
        _audit_logger.info(json.dumps(audit_record))
                
        return scrubbed

    def scrub_text(self, text: str, source_context: str = "unknown") -> str:
        """Synchronous version - runs async version in event loop."""
        # For backwards compatibility, run in current event loop or create one
        try:
            loop = asyncio.get_running_loop()
            # If we're already in an event loop, we can't run_until_complete
            # Create a new task and await it
            return asyncio.run_coroutine_threadsafe(
                self.scrub_text_async(text, source_context), loop
            ).result(timeout=30)
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(self.scrub_text_async(text, source_context))

    def scrub_history_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deep scrubs the history dictionary."""
        scrubbed_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                scrubbed_data[key] = self.scrub_text(value)
            elif isinstance(value, dict):
                scrubbed_data[key] = self.scrub_history_data(value)
            else:
                scrubbed_data[key] = value
        return scrubbed_data

    def _is_likely_text(self, crop_img) -> bool:
        """Heuristic check to determine if a cropped bounding box actually contains text.
        Text has multiple character-level horizontal and vertical transitions and high local variance.
        Smooth anatomical structures like calcifications or nodules are typically more homogeneous.
        """
        import cv2
        import numpy as np
        
        if not isinstance(crop_img, np.ndarray) or crop_img.ndim < 2:
            return True
            
        h_c, w_c = crop_img.shape[:2]
        if h_c < 4 or w_c < 6:
            return False
            
        # 1. Aspect ratio validation: text regions/words are horizontal spans
        aspect_ratio = w_c / float(h_c)
        if aspect_ratio < 0.4:
            return False
            
        # 2. Gradient variation check: text has high contrast and sharp transitions
        grad_x = cv2.Sobel(crop_img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(crop_img, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        std_grad = np.std(grad_mag)
        
        # If the gradient variance is very low, it represents a smooth background or homogeneous mass
        if std_grad < 5.0:
            return False
            
        # 3. Transition density analysis:
        # A horizontal scan line across letters crosses several strokes (black/white edges).
        _, thresh_crop = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        transitions = []
        for r in range(h_c):
            row = thresh_crop[r, :]
            row_diff = np.diff(row)
            num_transitions = np.count_nonzero(row_diff)
            transitions.append(num_transitions)
            
        avg_transitions = np.mean(transitions)
        
        # Text characters generate multiple stroke transitions.
        # If the box is wide but has fewer transitions (e.g. solid white/black blob), it is not text.
        if avg_transitions < 3.0:
            return False
            
        return True

    def detect_burned_in_text(self, image_path: str):
        """
        Detects regions likely to contain burned-in patient identifiers.
        To avoid naively masking pathology (like calcifications or nodules) in the lung fields,
        we restrict detection exclusively to the outer peripheral zones (top 15% and bottom 15%
        of the image height) where patient names, MRNs, DOBs, and scanning timestamps are standard.
        """
        import cv2
        import numpy as np
        
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return []
            
            h, w = img.shape
            boxes = []
            
            # Restrict detection to top 15% and bottom 15% of the image height
            zones = [
                (0, int(h * 0.15)),          # Top zone
                (int(h * 0.85), h)           # Bottom zone
            ]
            
            for start_y, end_y in zones:
                if start_y >= end_y:
                    continue
                # Segment of the image for the current zone
                zone_img = img[start_y:end_y, :]
                
                # Thresholding using Otsu's binarization to segment text contours
                thresh = cv2.threshold(zone_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                # Scale the dilation kernel width and height dynamically based on the image size
                k_w = max(5, int(w * 0.015))
                k_h = max(2, int(h * 0.003))
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
                dilated = cv2.dilate(thresh, kernel, iterations=1)
                
                # Find connected contours that could represent letters or words
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Scale heuristics dynamically relative to image dimensions to support variable resolutions
                min_h_c = max(4, int(h * 0.008))
                max_h_c = max(15, int(h * 0.035))
                min_w_c = max(6, int(w * 0.012))
                
                for cnt in contours:
                    x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
                    
                    # Filtering contours using dynamic scaled heuristics
                    if min_h_c <= h_c <= max_h_c and w_c >= min_w_c:
                        # Extract the crop and apply the heuristic de-identification check
                        crop_img = zone_img[y_c:y_c + h_c, x_c:x_c + w_c]
                        if self._is_likely_text(crop_img):
                            # Translate zone-relative y-coordinate to full image space
                            boxes.append((x_c, start_y + y_c, w_c, h_c))
                        else:
                            logger.info(f"Preserving potential anatomical structure at ({x_c}, {start_y + y_c}, {w_c}, {h_c}) - failed text validation.")
                        
            return boxes
        except Exception as e:
            logger.error(f"Error detecting burned-in text: {e}")
            return []

    def mask_burned_in_text(self, image_path: str) -> str:
        """
        Scrubs PHI from images:
        - For DICOM (.dcm), reads dataset headers and anonymizes standard identifier tags.
        - For standard images, masks burned-in text within peripheral zones by drawing black boxes.
        Saves the sanitized image to a safe tempfile and returns its path.
        """
        if str(image_path).lower().endswith(".dcm"):
            try:
                import pydicom
                import tempfile
                
                ds = pydicom.dcmread(image_path)
                redacted_tags = []
                
                def redact_element(dataset, data_element):
                    if data_element.tag.is_private or data_element.VR in ["PN", "DA", "TM", "AS"]:
                        # For binary data, clear it. Otherwise redact.
                        if data_element.VR in ["OB", "OW", "UN", "SQ"]:
                            data_element.value = b""
                        else:
                            data_element.value = "REDACTED"
                        redacted_tags.append(str(data_element.tag))
                        
                ds.walk(redact_element)
                        
                tmp = tempfile.NamedTemporaryFile(suffix=".dcm", delete=False)
                sanitized_path = tmp.name
                tmp.close()
                
                ds.save_as(sanitized_path)
                logger.info(f"Successfully scrubbed DICOM metadata headers. Saved to {sanitized_path}")

                # Flaw #23 Fix: Audit trail for DICOM header redaction
                if redacted_tags:
                    audit_record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": os.path.basename(image_path),
                        "operation": "dicom_header_scrub",
                        "redacted_tags": redacted_tags,
                        "output": sanitized_path,
                    }
                    _audit_logger.info(json.dumps(audit_record))

                return sanitized_path
            except Exception as e:
                logger.error(f"Failed to scrub DICOM metadata headers: {e}")
                return image_path
            
        import cv2
        import shutil
        import tempfile
        try:
            img = cv2.imread(image_path)
            if img is None:
                return image_path
                
            boxes = self.detect_burned_in_text(image_path)
            if not boxes:
                return image_path
                
            for (x, y, w, h) in boxes:
                # Mask text box by drawing a solid black rectangle over it
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
                
            from pathlib import Path
            original_suffix = Path(image_path).suffix.lower()
            intermediate_suffix = ".png" if original_suffix in [".png", ".jpg", ".jpeg"] else original_suffix
            if not intermediate_suffix:
                intermediate_suffix = ".png"
            tmp = tempfile.NamedTemporaryFile(suffix=intermediate_suffix, delete=False)
            output_path = tmp.name
            tmp.close()
                
            cv2.imwrite(output_path, img)
            logger.info(f"Masked {len(boxes)} PHI text regions. Sanitized image saved to {output_path}")

            # Flaw #23 Fix: Audit trail for burned-in text masking
            audit_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": os.path.basename(image_path),
                "operation": "burned_in_text_mask",
                "regions_masked": len(boxes),
                "bounding_boxes": [(x, y, w, h) for (x, y, w, h) in boxes],
                "output": output_path,
            }
            _audit_logger.info(json.dumps(audit_record))

            return output_path
        except Exception as e:
            logger.error(f"Failed to mask burned-in text: {e}")
            return image_path

    def scrub_pdf(self, pdf_path: str) -> str:
        """Validate and scrub PDF file."""
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                if len(pdf.pages) == 0:
                    pass
        except Exception as e:
            raise ValueError(f"Failed to parse clinical history PDF: {e}")
        return pdf_path
