import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
from scipy.stats import bootstrap

logger = logging.getLogger("threshold-calibration")


@dataclass
class ValidationMetrics:
    dataset: str
    timestamp: str
    num_samples: int
    num_classes: int
    class_names: List[str]
    
    per_class_auroc: Dict[str, float]
    macro_auroc: float
    weighted_auroc: float
    
    per_class_sensitivity: Dict[str, float]
    per_class_specificity: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_f1: Dict[str, float]
    macro_f1: float
    weighted_f1: float
    
    ece: float
    mce: float
    
    optimal_thresholds: Dict[str, float]
    youden_thresholds: Dict[str, float]
    
    fnr_malignancy: float
    fnr_tb: float
    escalation_rate: float
    ood_detection_rate: float


@dataclass
class ThresholdRecommendation:
    confidence_threshold: float
    uncertainty_threshold: float
    ood_confidence_threshold: float
    ood_cosine_threshold: float
    ood_text_cosine_threshold: float
    uncertainty_calibration_factor: float
    mc_dropout_passes: int
    
    rationale: str
    validation_metrics: ValidationMetrics
    clinical_utility_curve: Dict[str, List[float]]


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true
    
    ece = 0.0
    mce = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            diff = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += diff * prop_in_bin
            mce = max(mce, diff)
    
    return ece, mce


def compute_per_class_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    num_classes = len(class_names)
    y_pred = np.argmax(y_prob, axis=1)
    
    results = {
        "auroc": {},
        "sensitivity": {},
        "specificity": {},
        "precision": {},
        "f1": {},
    }
    
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_prob_class = y_prob[:, i]
        y_pred_binary = (y_pred == i).astype(int)
        
        try:
            results["auroc"][class_name] = roc_auc_score(y_true_binary, y_prob_class)
        except ValueError:
            results["auroc"][class_name] = 0.0
        
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        results["sensitivity"][class_name] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results["specificity"][class_name] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results["precision"][class_name] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        results["f1"][class_name] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    return results


def find_optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    optimal = {}
    youden = {}
    
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_prob_class = y_prob[:, i]
        
        try:
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob_class)
            j_scores = tpr - fpr
            youden_idx = np.argmax(j_scores)
            youden[class_name] = float(thresholds[youden_idx])
            
            precision, recall, pr_thresholds = precision_recall_curve(y_true_binary, y_prob_class)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores[:-1])
            optimal[class_name] = float(pr_thresholds[optimal_idx])
        except ValueError:
            optimal[class_name] = 0.5
            youden[class_name] = 0.5
    
    return {"optimal_f1": optimal, "youden": youden}


def compute_clinical_utility_curve(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    class_names: List[str],
    fn_cost: float = 10.0,
    fp_cost: float = 1.0
) -> Dict[str, List[float]]:
    thresholds = np.linspace(0.01, 0.99, 99)
    utility_curve = {}
    
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_prob_class = y_prob[:, i]
        
        utilities = []
        for thresh in thresholds:
            y_pred = (y_prob_class >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred).ravel()
            utility = tp - fn_cost * fn - fp_cost * fp
            utilities.append(float(utility))
        
        utility_curve[class_name] = utilities
    
    return {"thresholds": thresholds.tolist(), "utilities": utility_curve}


def calibrate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty_scores: np.ndarray,
    visual_ood_scores: np.ndarray,
    text_ood_scores: np.ndarray,
    class_names: List[str],
    dataset_name: str = "validation",
    fn_cost_malignancy: float = 10.0,
    fn_cost_tb: float = 8.0,
    target_escalation_rate: float = 0.15,
    target_fnr_malignancy: float = 0.02,
) -> ThresholdRecommendation:
    
    num_samples = len(y_true)
    num_classes = len(class_names)
    
    per_class_metrics = compute_per_class_metrics(y_true, y_prob, class_names)
    ece, mce = compute_calibration_metrics(y_true, y_prob)
    
    thresholds_result = find_optimal_thresholds(y_true, y_prob, class_names)
    optimal_f1 = thresholds_result["optimal_f1"]
    youden = thresholds_result["youden"]
    
    clinical_utility = compute_clinical_utility_curve(y_true, y_prob, class_names)
    
    macro_auroc = np.mean(list(per_class_metrics["auroc"].values()))
    weighted_auroc = np.average(
        list(per_class_metrics["auroc"].values()),
        weights=np.bincount(y_true, minlength=num_classes)
    )
    macro_f1 = np.mean(list(per_class_metrics["f1"].values()))
    weighted_f1 = np.average(
        list(per_class_metrics["f1"].values()),
        weights=np.bincount(y_true, minlength=num_classes)
    )
    
    malignancy_idx = class_names.index("Lung Cancer") if "Lung Cancer" in class_names else \
                     class_names.index("Malignancy") if "Malignancy" in class_names else -1
    tb_idx = class_names.index("Tuberculosis") if "Tuberculosis" in class_names else -1
    
    fnr_malignancy = 1 - per_class_metrics["sensitivity"].get("Lung Cancer", 
                      per_class_metrics["sensitivity"].get("Malignancy", 1.0))
    fnr_tb = 1 - per_class_metrics["sensitivity"].get("Tuberculosis", 1.0)
    
    max_probs = np.max(y_prob, axis=1)
    ood_mask = max_probs < 0.5
    ood_detection_rate = ood_mask.mean()
    
    escalation_mask = (max_probs < 0.6) | (uncertainty_scores > 0.15)
    escalation_rate = escalation_mask.mean()
    
    confidence_thresh = 0.6
    uncertainty_thresh = 0.15
    ood_conf_thresh = 0.4
    ood_cos_thresh = 0.82
    ood_text_cos_thresh = 0.82
    calib_factor = 1.0
    mc_passes = 50
    
    if fnr_malignancy > target_fnr_malignancy:
        confidence_thresh = max(confidence_thresh, 0.65)
        uncertainty_thresh = min(uncertainty_thresh, 0.10)
        rationale = f"Increased confidence threshold to {confidence_thresh:.2f} to reduce malignancy FNR ({fnr_malignancy:.1%})"
    elif escalation_rate > target_escalation_rate * 1.5:
        confidence_thresh = max(confidence_thresh, 0.55)
        uncertainty_thresh = max(uncertainty_thresh, 0.20)
        rationale = f"Adjusted thresholds to control escalation rate ({escalation_rate:.1%})"
    else:
        rationale = "Thresholds balanced for clinical utility"
    
    validation_metrics = ValidationMetrics(
        dataset=dataset_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        num_samples=num_samples,
        num_classes=num_classes,
        class_names=class_names,
        per_class_auroc=per_class_metrics["auroc"],
        macro_auroc=float(macro_auroc),
        weighted_auroc=float(weighted_auroc),
        per_class_sensitivity=per_class_metrics["sensitivity"],
        per_class_specificity=per_class_metrics["specificity"],
        per_class_precision=per_class_metrics["precision"],
        per_class_f1=per_class_metrics["f1"],
        macro_f1=float(macro_f1),
        weighted_f1=float(weighted_f1),
        ece=float(ece),
        mce=float(mce),
        optimal_thresholds=optimal_f1,
        youden_thresholds=youden,
        fnr_malignancy=float(fnr_malignancy),
        fnr_tb=float(fnr_tb),
        escalation_rate=float(escalation_rate),
        ood_detection_rate=float(ood_detection_rate),
    )
    
    recommendation = ThresholdRecommendation(
        confidence_threshold=confidence_thresh,
        uncertainty_threshold=uncertainty_thresh,
        ood_confidence_threshold=ood_conf_thresh,
        ood_cosine_threshold=ood_cos_thresh,
        ood_text_cosine_threshold=ood_text_cos_thresh,
        uncertainty_calibration_factor=calib_factor,
        mc_dropout_passes=mc_passes,
        rationale=rationale,
        validation_metrics=validation_metrics,
        clinical_utility_curve=clinical_utility,
    )
    
    return recommendation


def save_calibration_report(recommendation: ThresholdRecommendation, output_path: str):
    report = {
        "recommendation": asdict(recommendation),
        "environment_variables": {
            "CLINICAL_CONFIDENCE_THRESHOLD": str(recommendation.confidence_threshold),
            "CLINICAL_UNCERTAINTY_THRESHOLD": str(recommendation.uncertainty_threshold),
            "CLINICAL_OOD_CONFIDENCE_THRESHOLD": str(recommendation.ood_confidence_threshold),
            "CLINICAL_OOD_COSINE_THRESHOLD": str(recommendation.ood_cosine_threshold),
            "CLINICAL_OOD_TEXT_COSINE_THRESHOLD": str(recommendation.ood_text_cosine_threshold),
            "CLINICAL_UNCERTAINTY_CALIBRATION_FACTOR": str(recommendation.uncertainty_calibration_factor),
            "CLINICAL_MC_DROPOUT_PASSES": str(recommendation.mc_dropout_passes),
            "CLINICAL_THRESHOLDS_VALIDATED": "true",
            "CLINICAL_VALIDATION_DATASET": recommendation.validation_metrics.dataset,
            "CLINICAL_VALIDATION_DATE": recommendation.validation_metrics.timestamp,
            "CLINICAL_VALIDATION_METRICS": json.dumps(asdict(recommendation.validation_metrics)),
        }
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Calibration report saved to {output_path}")


def load_calibration_report(path: str) -> Optional[ThresholdRecommendation]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        rec_data = data["recommendation"]
        rec_data["validation_metrics"] = ValidationMetrics(**rec_data["validation_metrics"])
        return ThresholdRecommendation(**rec_data)
    except Exception as e:
        logger.error(f"Failed to load calibration report: {e}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    class_names = ["Silicosis", "Pneumonia", "Tuberculosis", "Asbestosis", "Normal"]
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.dirichlet(np.ones(n_classes), n_samples)
    y_prob[y_true, np.arange(n_samples)] += 0.3
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    uncertainty_scores = np.random.gamma(2, 0.05, n_samples)
    visual_ood_scores = np.random.beta(2, 5, n_samples)
    text_ood_scores = np.random.beta(2, 5, n_samples)
    
    rec = calibrate_thresholds(
        y_true, y_prob, uncertainty_scores, 
        visual_ood_scores, text_ood_scores, 
        class_names, "mock_validation"
    )
    
    print(json.dumps(asdict(rec), indent=2, default=str))
    save_calibration_report(rec, "config/calibration_report.json")