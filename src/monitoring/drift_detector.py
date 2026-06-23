import numpy as np
import json
from pathlib import Path
from scipy.stats import ks_2samp
import logging
import redis
import os
import requests as http_requests
from datetime import datetime, timezone
import asyncio

logger = logging.getLogger("drift-detector")

# Flaw #16 Fix: Configurable alert webhook for drift notifications
DRIFT_ALERT_WEBHOOK_URL = os.getenv("DRIFT_ALERT_WEBHOOK_URL", "")

# Flaw #10 Fix: TTL for Redis drift window keys (default: 24 hours)
DRIFT_KEY_TTL_SECONDS = int(os.getenv("DRIFT_KEY_TTL_SECONDS", "86400"))

# Configurable minimum cases and threshold for concept/performance drift alerts to prevent alert fatigue
# Tighter thresholds to safeguard patients and ensure quick alerting upon model performance decay.
DRIFT_MIN_CASES = int(os.getenv("DRIFT_MIN_CASES", "50"))
DRIFT_AGREEMENT_THRESHOLD = float(os.getenv("DRIFT_AGREEMENT_THRESHOLD", "0.95"))

import concurrent.futures
_alert_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="drift-alert-sender")

def _send_alert_sync(title: str, message: str):
    logger.critical(f"DRIFT ALERT: {title} — {message}")
    if DRIFT_ALERT_WEBHOOK_URL:
        try:
            payload = {
                "text": f"🚨 *{title}*\n{message}\n_Timestamp: {datetime.now(timezone.utc).isoformat()}_"
            }
            http_requests.post(DRIFT_ALERT_WEBHOOK_URL, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send drift alert webhook: {e}")

def _send_alert(title: str, message: str):
    """Flaw #2 Fix: Send drift alert asynchronously via a thread pool to avoid blocking execution."""
    _alert_executor.submit(_send_alert_sync, title, message)


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

        # Lua script to atomically push elements, set TTL, check count, and conditionally pop both windows
        self.lua_push_and_check = """
        local window_key = KEYS[1]
        local features_key = KEYS[2]
        local val_probs = ARGV[1]
        local val_features = ARGV[2]
        local ttl = tonumber(ARGV[3])
        local threshold = tonumber(ARGV[4])
        
        -- Push probabilities
        redis.call('RPUSH', window_key, val_probs)
        redis.call('EXPIRE', window_key, ttl)
        
        -- Push features if provided
        if val_features ~= "" then
            redis.call('RPUSH', features_key, val_features)
            redis.call('EXPIRE', features_key, ttl)
        end
        
        -- Check size
        local count = redis.call('LLEN', window_key)
        if count >= threshold then
            local probs = redis.call('LRANGE', window_key, 0, -1)
            local features = redis.call('LRANGE', features_key, 0, -1)
            redis.call('DEL', window_key)
            redis.call('DEL', features_key)
            return {probs, features}
        end
        return nil
        """

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

    async def add_prediction(self, probs: list, visual_features: list = None):
        """Adds current prediction probabilities and visual features to persistent distributed windows.
        Offloads Redis operations and CPU-bound statistical tests to a separate thread pool."""
        if self.disabled:
            return

        try:
            await asyncio.to_thread(self._add_prediction_sync, probs, visual_features)
        except Exception as e:
            logger.error(f"Failed to monitor drift: {e}")

    def _add_prediction_sync(self, probs: list, visual_features: list = None):
        """Synchronous implementation of add_prediction to be run in asyncio.to_thread."""
        try:
            val_probs = json.dumps(probs)
            val_features = json.dumps(visual_features) if visual_features is not None else ""
            
            # Execute Lua script to push and conditionally retrieve expired lists atomically
            result = self.redis_client.eval(
                self.lua_push_and_check,
                2,
                self.window_key,
                self.features_window_key,
                val_probs,
                val_features,
                DRIFT_KEY_TTL_SECONDS,
                100
            )
            
            if result:
                raw_probs, raw_features = result
                current_probs = [json.loads(p) for p in raw_probs]
                current_features = [json.loads(f) for f in raw_features] if (raw_features and len(raw_features) > 0) else None
                
                # Perform drift calculations sequentially in this background thread
                self.check_prediction_drift(current_probs)
                if current_features and len(current_features) > 0:
                    self.check_covariate_shift(current_features)
                self.check_performance_drift()
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}. Cannot monitor drift.")
        except Exception as e:
            logger.error(f"Error executing drift checks: {e}")

    def check_prediction_drift(self, current_probs: list):
        """
        Monitors Prediction Drift / Label Shift P(Y_hat) using Kolmogorov-Smirnov test.
        """
        if self.disabled or not current_probs:
            return False

        if self.baseline is None:
            logger.warning("No prediction baseline found for drift detection. Saving current as baseline.")
            self._save_baseline(current_probs)
            return False

        current = np.array(current_probs)
        
        # Validate prediction dimensions (prevent IndexError on class mismatch)
        if current.shape[1] != self.baseline.shape[1]:
            msg = (
                f"Prediction class count mismatch: baseline={self.baseline.shape[1]}, "
                f"current={current.shape[1]}. Baseline will NOT be reset. Manual rollback / intervention required."
            )
            _send_alert("Model Architecture Mismatch", msg)
            return False

        drift_detected = False
        
        # Compare distributions using Kolmogorov-Smirnov test per class
        for i in range(current.shape[1]):
            stat, p_value = ks_2samp(self.baseline[:, i], current[:, i])
            if p_value < 0.05:
                msg = f"Significant Prediction Drift (Label Shift) detected in Class {i} (p={p_value:.4f})"
                _send_alert("Prediction Drift", msg)
                drift_detected = True
        
        return drift_detected

    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
        """Linear-time unbiased MMD² estimator (Gretton et al. 2012, Section 6).

        Panel Flaw #5 Fix: The previous O(N²) implementation computed full
        pairwise distance matrices (N×N, N×M, M×M) which blocked the Python
        GIL for the entire computation, freezing concurrent async workers.

        This estimator pairs samples 1:1 and computes kernel differences in
        O(N) time using the h-statistic:
            h_i = k(x_{2i}, x_{2i+1}) + k(y_{2i}, y_{2i+1})
                - k(x_{2i}, y_{2i+1}) - k(x_{2i+1}, y_{2i})
            MMD² ≈ (1/m) Σ h_i   where m = floor(N/2)

        This is unbiased and consistent, with variance O(1/N) vs O(1/N²) for
        the quadratic estimator — a modest statistical trade-off for a massive
        computational speedup under production traffic.
        """
        n = min(X.shape[0], Y.shape[0])
        if n < 4:
            return 0.0

        # Truncate to equal length and shuffle for unbiased pairing
        X, Y = X[:n], Y[:n]
        perm = np.random.permutation(n)
        X, Y = X[perm], Y[perm]

        # Use only even number of samples for clean pairing
        m = n // 2
        X_even, X_odd = X[:m], X[m:2*m]
        Y_even, Y_odd = Y[:m], Y[m:2*m]

        # Estimate gamma from a small subsample if not provided
        if gamma is None:
            subsample_size = min(m, 50)
            dists = np.sum((X_even[:subsample_size] - Y_even[:subsample_size]) ** 2, axis=1)
            median_dist = np.median(dists)
            gamma = 1.0 / (median_dist + 1e-8)

        def rbf_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """Vectorized RBF kernel for paired rows: k(a_i, b_i) for all i."""
            sq_dists = np.sum((a - b) ** 2, axis=1)
            return np.exp(-gamma * sq_dists)

        # h_i = k(x_even, x_odd) + k(y_even, y_odd) - k(x_even, y_odd) - k(x_odd, y_even)
        h = (rbf_kernel(X_even, X_odd)
             + rbf_kernel(Y_even, Y_odd)
             - rbf_kernel(X_even, Y_odd)
             - rbf_kernel(X_odd, Y_even))

        return float(np.mean(h))

    def check_covariate_shift(self, current_features: list):
        """
        Monitors Covariate Shift P(X) on visual features.
        Compares visual embeddings of the current window against a baseline using MMD and cosine similarity.
        """
        if self.disabled or not current_features:
            return False

        if self.features_baseline is None:
            logger.warning("No visual features baseline found. Saving current window as baseline.")
            self._save_features_baseline(current_features)
            return False

        # Validate feature dimensions before creating numpy array
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
            
        # 1. Compute MMD (Maximum Mean Discrepancy) - First-principles multidimensional feature shift
        mmd_value = self._compute_mmd(self.features_baseline, current)
        logger.info(f"Visual covariate shift analysis - Maximum Mean Discrepancy (MMD): {mmd_value:.6f}")

        # 2. Compute Cosine Similarity between baseline and current mean feature vectors
        mean_baseline = np.mean(self.features_baseline, axis=0)
        mean_current = np.mean(current, axis=0)
        
        norm_b = np.linalg.norm(mean_baseline)
        norm_c = np.linalg.norm(mean_current)
        
        cosine_sim = 1.0
        if norm_b > 0 and norm_c > 0:
            cosine_sim = np.dot(mean_baseline, mean_current) / (norm_b * norm_c)
            logger.info(f"Visual covariate shift analysis - Cosine Similarity: {cosine_sim:.4f}")
        
        # Alert if MMD exceeds threshold (indicating distribution shift) or cosine similarity falls below threshold
        mmd_threshold = 0.05
        cosine_threshold = 0.95
        
        if mmd_value > mmd_threshold or cosine_sim < cosine_threshold:
            msg = (
                f"Significant Covariate Shift P(X) detected in visual feature space! "
                f"MMD: {mmd_value:.6f} (threshold: {mmd_threshold}), "
                f"Cosine Similarity: {cosine_sim:.4f} (threshold: {cosine_threshold})."
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
