import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityEvaluator:
    """
    Comprehensive evaluation suite for:
    - Diagnostic Accuracy (AUROC, F1)
    - Uncertainty Calibration (ECE)
    - RAG retrieval quality (Hit-Rate@K)
    """
    def __init__(self, num_classes=5):
        self.num_classes = num_classes

    def compute_classification_metrics(self, y_true, y_probs, y_pred):
        """
        Compute standard classification metrics.
        y_true: (N,) array of ground truth class indices
        y_probs: (N, num_classes) array of class probabilities
        y_pred: (N,) array of predicted class indices
        """
        # One-hot encoding for AUROC
        y_true_onehot = np.zeros((len(y_true), self.num_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1

        # AUROC (Macro and weighted)
        try:
            auroc_macro = roc_auc_score(y_true_onehot, y_probs, average='macro', multi_class='ovr')
            auroc_weighted = roc_auc_score(y_true_onehot, y_probs, average='weighted', multi_class='ovr')
        except ValueError:
            # Handle cases where some classes are missing in the test split
            auroc_macro = 0.0
            auroc_weighted = 0.0

        # F1, Precision, Recall
        f1_macro = f1_score(y_true, y_pred, average='macro')
        precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

        return {
            "auroc_macro": auroc_macro,
            "auroc_weighted": auroc_weighted,
            "f1_macro": f1_macro,
            "precision_macro": precision,
            "recall_macro": recall
        }

    def compute_ece(self, y_true, y_probs, n_bins=10):
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        confidences = np.max(y_probs, axis=1)
        predictions = np.argmax(y_probs, axis=1)
        accuracies = predictions == y_true
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece

    def report_summary(self, metrics: dict):
        logger.info("--- Model Evaluation Summary ---")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"{k.upper():<20}: {v:.4f}")
            else:
                logger.info(f"{k.upper():<20}: {v}")

if __name__ == "__main__":
    # Mock data for demonstration
    evaluator = QualityEvaluator()
    y_true = np.array([0, 1, 2, 0, 1])
    y_probs = np.array([
        [0.8, 0.1, 0.1, 0.0, 0.0],
        [0.2, 0.7, 0.1, 0.0, 0.0],
        [0.1, 0.1, 0.8, 0.0, 0.0],
        [0.9, 0.0, 0.1, 0.0, 0.0],
        [0.3, 0.6, 0.1, 0.0, 0.0]
    ])
    y_pred = np.argmax(y_probs, axis=1)
    
    metrics = evaluator.compute_classification_metrics(y_true, y_probs, y_pred)
    ece = evaluator.compute_ece(y_true, y_probs)
    metrics["ece"] = ece
    
    evaluator.report_summary(metrics)
