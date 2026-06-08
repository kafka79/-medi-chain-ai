import numpy as np
import json
from pathlib import Path
from scipy.stats import ks_2samp
import logging
import redis
import os
from datetime import datetime, timezone

logger = logging.getLogger("drift-detector")

class DriftDetector:
    """
    Monitors three tiers of model and data drift:
    1. Prediction Drift / Label Shift P(Y_hat) using KS-test on probability vectors.
    2. Covariate Shift P(X) on visual embeddings using cosine similarity.
    3. Performance Drift (True Concept Drift) P(Y|X) using clinician feedback.
    """
    def __init__(self):
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
        self.window_key = "medi_chain:drift:window"
        self.baseline_key = "medi_chain:drift:baseline"
        self.features_window_key = "medi_chain:drift:features_window"
        self.features_baseline_key = "medi_chain:drift:features_baseline"
        
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
        try:
            pipe = self.redis_client.pipeline()
            pipe.rpush(self.window_key, json.dumps(probs))
            if visual_features is not None:
                pipe.rpush(self.features_window_key, json.dumps(visual_features))
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
                logger.error(f"ALERT: Significant Prediction Drift (Label Shift) detected in Class {i} (p={p_value:.4f})")
                drift_detected = True
        
        return drift_detected

    def check_covariate_shift(self):
        """
        Monitors Covariate Shift P(X) on visual features.
        Compares visual embeddings of the current window against a baseline using cosine similarity.
        """
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

        current = np.array(current_features)
        
        # Ensure dimensions match
        if current.ndim != 2 or self.features_baseline.ndim != 2:
            logger.warning("Feature matrices dimensions are mismatched or invalid.")
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
            logger.error(
                f"ALERT: Significant Covariate Shift P(X) detected in visual feature space! "
                f"Cosine similarity to baseline is {cosine_sim:.4f} (threshold: 0.95)."
            )
            return True
            
        return False

    def check_performance_drift(self):
        """
        Monitors True Concept Drift P(Y|X) using clinician feedback.
        Loads historical feedback logs to check if the agreement rate has degraded.
        """
        try:
            from src.data.feedback_manager import FeedbackManager
            fm = FeedbackManager()
            summary_file = fm.summary_file
            if summary_file.exists():
                with open(summary_file, "r") as f:
                    summary = json.load(f)
                
                total = summary.get("total_cases", 0)
                rate = summary.get("agreement_rate", 1.0)
                
                # Check performance drift if we have at least 20 cases recorded
                if total >= 20 and rate < 0.80:
                    logger.critical(
                        f"ALERT: Performance Drift (True Concept Drift) detected! "
                        f"Clinician agreement rate dropped to {rate:.1%} across {total} cases."
                    )
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to check performance drift: {e}")
            return False
