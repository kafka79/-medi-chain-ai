import numpy as np
import json
from pathlib import Path
from scipy.stats import ks_2samp
import logging
import redis
import os
import requests as http_requests
from datetime import datetime, timezone

logger = logging.getLogger("drift-detector")

# Flaw #16 Fix: Configurable alert webhook for drift notifications
DRIFT_ALERT_WEBHOOK_URL = os.getenv("DRIFT_ALERT_WEBHOOK_URL", "")

# Flaw #10 Fix: TTL for Redis drift window keys (default: 24 hours)
DRIFT_KEY_TTL_SECONDS = int(os.getenv("DRIFT_KEY_TTL_SECONDS", "86400"))

# Configurable minimum cases and threshold for concept/performance drift alerts to prevent alert fatigue
DRIFT_MIN_CASES = int(os.getenv("DRIFT_MIN_CASES", "100"))
DRIFT_AGREEMENT_THRESHOLD = float(os.getenv("DRIFT_AGREEMENT_THRESHOLD", "0.80"))


def _send_alert(title: str, message: str):
    """Flaw #16 Fix: Send drift alert to external webhook (Slack, PagerDuty, etc.)."""
    logger.critical(f"DRIFT ALERT: {title} — {message}")
    if DRIFT_ALERT_WEBHOOK_URL:
        try:
            payload = {
                "text": f"🚨 *{title}*\n{message}\n_Timestamp: {datetime.now(timezone.utc).isoformat()}_"
            }
            http_requests.post(DRIFT_ALERT_WEBHOOK_URL, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send drift alert webhook: {e}")


class DriftDetector:
    """
    Monitors three tiers of model and data drift:
    1. Prediction Drift / Label Shift P(Y_hat) using KS-test on probability vectors.
    2. Covariate Shift P(X) on visual embeddings using cosine similarity.
    3. Performance Drift (True Concept Drift) P(Y|X) using clinician feedback.
    """
    def __init__(self):
        self.window_key = "medi_chain:drift:window"
        self.baseline_key = "medi_chain:drift:baseline"
        self.features_window_key = "medi_chain:drift:features_window"
        self.features_baseline_key = "medi_chain:drift:features_baseline"
        # Flaw #9 Fix: Performance drift feedback stored in Redis, not local disk
        self.feedback_summary_key = "medi_chain:drift:feedback_summary"
        self.disabled = os.getenv("TESTING") == "true"
        self.redis_client = None
        self.baseline = None
        self.features_baseline = None

        if self.disabled:
            return

        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_client = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            db=0, 
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=2
        )
        
        self.baseline = self._load_baseline()
        self.features_baseline = self._load_features_baseline()

    def _load_baseline(self):
        try:
            data = self.redis_client.get(self.baseline_key)
            if data:
                return np.array(json.loads(data))
        except Exception as e:
            logger.error(f"Failed to load prediction baseline from Redis: {e}")
        return None

    def _save_baseline(self, probs: list):
        try:
            self.redis_client.set(self.baseline_key, json.dumps(probs))
            self.baseline = np.array(probs)
            logger.info("Saved new global drift prediction baseline to Redis.")
        except Exception as e:
            logger.error(f"Failed to save prediction baseline to Redis: {e}")

    def _load_features_baseline(self):
        try:
            data = self.redis_client.get(self.features_baseline_key)
            if data:
                return np.array(json.loads(data))
        except Exception as e:
            logger.error(f"Failed to load features baseline from Redis: {e}")
        return None

    def _save_features_baseline(self, features: list):
        try:
            self.redis_client.set(self.features_baseline_key, json.dumps(features))
            self.features_baseline = np.array(features)
            logger.info("Saved new global features baseline (covariate shift) to Redis.")
        except Exception as e:
            logger.error(f"Failed to save features baseline to Redis: {e}")

    def add_prediction(self, probs: list, visual_features: list = None):
        """Adds current prediction probabilities and visual features to persistent distributed windows."""
        if self.disabled:
            return

        try:
            pipe = self.redis_client.pipeline()
            pipe.rpush(self.window_key, json.dumps(probs))
            # Flaw #10 Fix: Set TTL on drift window keys to prevent stale data accumulation
            pipe.expire(self.window_key, DRIFT_KEY_TTL_SECONDS)
            if visual_features is not None:
                pipe.rpush(self.features_window_key, json.dumps(visual_features))
                pipe.expire(self.features_window_key, DRIFT_KEY_TTL_SECONDS)
            pipe.llen(self.window_key)
            results = pipe.execute()
            
            count = results[-1]
            if count >= 100:  # Window size triggered
                self.check_prediction_drift()
                if visual_features is not None:
                    self.check_covariate_shift()
                self.check_performance_drift()
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}. Cannot monitor drift.")

    def check_prediction_drift(self):
        """
        Monitors Prediction Drift / Label Shift P(Y_hat) using Kolmogorov-Smirnov test.
        """
        if self.disabled:
            return False

        try:
            lua_script = """
            local data = redis.call('LRANGE', KEYS[1], 0, -1)
            redis.call('DEL', KEYS[1])
            return data
            """
            raw_probs = self.redis_client.eval(lua_script, 1, self.window_key)
            if not raw_probs:
                return False
            current_probs = [json.loads(p) for p in raw_probs]
        except redis.ConnectionError:
            return False

        if self.baseline is None:
            logger.warning("No prediction baseline found for drift detection. Saving current as baseline.")
            self._save_baseline(current_probs)
            return False

        current = np.array(current_probs)
        drift_detected = False
        
        # Compare distributions using Kolmogorov-Smirnov test per class
        for i in range(current.shape[1]):
            stat, p_value = ks_2samp(self.baseline[:, i], current[:, i])
            if p_value < 0.05:
                msg = f"Significant Prediction Drift (Label Shift) detected in Class {i} (p={p_value:.4f})"
                _send_alert("Prediction Drift", msg)
                drift_detected = True
        
        return drift_detected

    def check_covariate_shift(self):
        """
        Monitors Covariate Shift P(X) on visual features.
        Compares visual embeddings of the current window against a baseline using cosine similarity.
        """
        if self.disabled:
            return False

        try:
            lua_script = """
            local data = redis.call('LRANGE', KEYS[1], 0, -1)
            redis.call('DEL', KEYS[1])
            return data
            """
            raw_features = self.redis_client.eval(lua_script, 1, self.features_window_key)
            if not raw_features:
                return False
            
            current_features = [json.loads(f) for f in raw_features]
        except redis.ConnectionError:
            return False

        if self.features_baseline is None:
            logger.warning("No visual features baseline found. Saving current window as baseline.")
            self._save_features_baseline(current_features)
            return False

        # Flaw #12 Fix: Validate feature dimensions before creating numpy array
        current = np.array(current_features)
        if current.dtype == object:
            logger.error(
                "Feature vectors have inconsistent dimensions (dtype=object). "
                "This likely indicates a model version change. Resetting baseline."
            )
            self._save_features_baseline(current_features)
            return False
        
        # Ensure dimensions match
        if current.ndim != 2 or self.features_baseline.ndim != 2:
            logger.warning("Feature matrices dimensions are mismatched or invalid.")
            return False

        if current.shape[1] != self.features_baseline.shape[1]:
            logger.error(
                f"Feature dimension mismatch: baseline={self.features_baseline.shape[1]}, "
                f"current={current.shape[1]}. Resetting baseline to current."
            )
            self._save_features_baseline(current_features)
            return False
            
        # Compute mean feature embeddings
        mean_baseline = np.mean(self.features_baseline, axis=0)
        mean_current = np.mean(current, axis=0)
        
        # Calculate Cosine Similarity between baseline and current mean feature vectors
        norm_b = np.linalg.norm(mean_baseline)
        norm_c = np.linalg.norm(mean_current)
        
        if norm_b == 0 or norm_c == 0:
            return False
            
        cosine_sim = np.dot(mean_baseline, mean_current) / (norm_b * norm_c)
        logger.info(f"Visual covariate shift analysis - Cosine Similarity: {cosine_sim:.4f}")
        
        # A drop in cosine similarity indicates visual covariate shift (e.g., scanner change, noise)
        if cosine_sim < 0.95:
            msg = (
                f"Significant Covariate Shift P(X) detected in visual feature space! "
                f"Cosine similarity to baseline is {cosine_sim:.4f} (threshold: 0.95)."
            )
            _send_alert("Covariate Shift", msg)
            return True
            
        return False

    def check_performance_drift(self):
        """
        Monitors True Concept Drift P(Y|X) using clinician feedback.
        Flaw #9 Fix: Reads from Redis instead of local disk to work correctly across replicas.
        """
        if self.disabled:
            return False

        try:
            summary_raw = self.redis_client.get(self.feedback_summary_key)
            if not summary_raw:
                return False
            
            summary = json.loads(summary_raw)
            total = summary.get("total_cases", 0)
            rate = summary.get("agreement_rate", 1.0)
            
            # Check performance drift if we have at least DRIFT_MIN_CASES recorded
            if total >= DRIFT_MIN_CASES and rate < DRIFT_AGREEMENT_THRESHOLD:
                msg = (
                    f"Clinician agreement rate dropped to {rate:.1%} across {total} cases (threshold: {DRIFT_AGREEMENT_THRESHOLD:.1%})."
                )
                _send_alert("Performance Drift (Concept Drift)", msg)
                return True
            return False
        except redis.ConnectionError as e:
            logger.error(f"Failed to check performance drift — Redis unavailable: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to check performance drift: {e}")
            return False

    def update_feedback_summary(self, agreement: bool):
        """
        Flaw #9 Fix: Atomically update feedback summary in Redis so all replicas share the same data.
        Called by FeedbackManager when new feedback is received.
        """
        if self.disabled:
            return

        try:
            lua_script = """
            local key = KEYS[1]
            local is_agreement = ARGV[1] == "1"
            local raw = redis.call('GET', key)
            local summary = {
                ["total_cases"] = 0,
                ["agreements"] = 0,
                ["disagreements"] = 0,
                ["agreement_rate"] = 1.0
            }
            if raw and raw ~= false then
                summary = cjson.decode(raw)
            end
            summary["total_cases"] = tonumber(summary["total_cases"]) or 0
            summary["agreements"] = tonumber(summary["agreements"]) or 0
            summary["disagreements"] = tonumber(summary["disagreements"]) or 0
            summary["total_cases"] = summary["total_cases"] + 1
            if is_agreement then
                summary["agreements"] = summary["agreements"] + 1
            else
                summary["disagreements"] = summary["disagreements"] + 1
            end
            summary["agreement_rate"] = summary["agreements"] / summary["total_cases"]
            redis.call('SET', key, cjson.encode(summary))
            return cjson.encode(summary)
            """
            self.redis_client.eval(lua_script, 1, self.feedback_summary_key, "1" if agreement else "0")
        except Exception as e:
            logger.error(f"Failed to update feedback summary in Redis: {e}")
