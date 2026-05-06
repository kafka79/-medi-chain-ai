import numpy as np
import json
from pathlib import Path
from scipy.stats import ks_2samp
import logging

logger = logging.getLogger("drift-detector")

class DriftDetector:
    """
    Addresses the 'Model Drifting' flaw.
    Monitors prediction probability distributions against a 'Golden Baseline'.
    """
    def __init__(self, baseline_path: str = "configs/baseline_distribution.json"):
        self.baseline_path = Path(baseline_path)
        self.current_probs = []
        self.baseline = self._load_baseline()

    def _load_baseline(self):
        if self.baseline_path.exists():
            with open(self.baseline_path, "r") as f:
                return np.array(json.load(f))
        return None

    def add_prediction(self, probs: list):
        """Adds current prediction probabilities to the window."""
        self.current_probs.append(probs)
        if len(self.current_probs) > 100: # Window size
            self.check_for_drift()

    def check_for_drift(self):
        if self.baseline is None:
            logger.warning("No baseline found for drift detection. Saving current as baseline.")
            self._save_baseline()
            return False

        current = np.array(self.current_probs)
        # Compare distributions using Kolmogorov-Smirnov test
        for i in range(current.shape[1]): # Per class
            stat, p_value = ks_2samp(self.baseline[:, i], current[:, i])
            if p_value < 0.05:
                logger.error(f"ALERT: Significant Concept Drift detected in Class {i} (p={p_value:.4f})")
                return True
        
        # Reset window after check
        self.current_probs = []
        return False

    def _save_baseline(self):
        if self.current_probs:
            with open(self.baseline_path, "w") as f:
                json.dump(self.current_probs, f)
            self.baseline = np.array(self.current_probs)
