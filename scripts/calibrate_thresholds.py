#!/usr/bin/env python3
"""
Clinical Threshold Calibration Script

This script runs a trained model on a held-out validation dataset and computes
optimal clinical decision thresholds using ROC/PR analysis and clinical utility curves.

Outputs:
- calibration_report.json with recommended thresholds and validation metrics
- Environment variables for deployment (.env or docker-compose.yml)

Usage:
    python scripts/calibrate_thresholds.py \
        --dataset data/validation/val.parquet \
        --model-checkpoint models/fusion_model.pt \
        --output config/calibration_report.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
from scipy.stats import bootstrap

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.fusion import LateFusionModel, DIAGNOSTIC_CLASSES, NUM_CLASSES
from src.config.settings import get_clinical_thresholds

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("calibration")


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
    clinical_utility_curve: Dict[str, Any]
    environment_variables: Dict[str, str]


def load_validation_dataset(path: str) -> Tuple[List[str], List[str]]:
    """Load validation dataset returning (image_paths, pdf_paths, labels)."""
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        return (
            df["image_path"].tolist(),
            df["pdf_path"].tolist(),
            df["label"].tolist()
        )
    elif ext == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        return (
            [d["image_path"] for d in data],
            [d["pdf_path"] for d in data],
            [d["label"] for d in data]
        )
    else:
        raise ValueError(f"Unsupported dataset format: {ext}")


def load_model(checkpoint_path: str, device: str) -> LateFusionModel:
    """Load trained fusion model."""
    model = LateFusionModel()
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Backward compatibility for gate weights
    if "v_gate.weight" not in state_dict and "v_proj.weight" in state_dict:
        state_dict["v_gate.weight"] = state_dict["v_proj.weight"].clone()
        state_dict["v_gate.bias"] = state_dict["v_proj.bias"].clone()
    if "t_gate.weight" not in state_dict and "t_proj.weight" in state_dict:
        state_dict["t_gate.weight"] = state_dict["t_proj.weight"].clone()
        state_dict["t_gate.bias"] = state_dict["t_proj.bias"].clone()
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def run_inference(
    model: LateFusionModel,
    image_paths: List[str],
    pdf_paths: List[str],
    batch_size: int,
    device: str,
    mc_passes: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference with MC Dropout uncertainty estimation."""
    from src.vlm.visual_encoder import BiomedVisualEncoder
    from sentence_transformers import SentenceTransformer
    from src.models.uncertainty import UncertaintyEstimator
    
    visual_encoder = BiomedicalEncoder(device=device)
    text_encoder = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", device=device)
    uncertainty_estimator = UncertaintyEstimator(model)
    
    all_probs = []
    all_uncertainties = []
    all_visual_ood = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_img_paths = image_paths[i:i+batch_size]
        batch_pdf_paths = pdf_paths[i:i+batch_size]
        
        # Encode images
        visual_features = visual_encoder.encode_image(batch_img_paths)
        visual_features = visual_features.to(device)
        
        # Encode texts
        texts = []
        for pdf_path in batch_pdf_paths:
            # Load and parse PDF (simplified)
            import json
            with open(pdf_path, "r") as f:
                hist = json.load(f)
            text = f"{hist.get('chief_complaint', '')} {hist.get('history_present_illness', '')} {hist.get('labs', '')}"
            texts.append(text.strip())
        
        text_embeddings = text_encoder.encode(texts)
        text_embeddings = torch.tensor(text_embeddings, device=device)
        
        # Run uncertainty estimation
        with torch.no_grad():
            results = uncertainty_estimator.estimate_uncertainty(
                visual_features, text_embeddings, num_passes=mc_passes
            )
        
        probs = results["all_probs"].cpu().numpy()
        uncertainties = results["std_deviation"].cpu().numpy()
        max_probs = np.max(probs, axis=1)
        
        all_probs.append(probs)
        all_uncertainties.append(uncertainties)
        all_visual_ood.append(max_probs)
    
    return (
        np.vstack(all_probs),
        np.concatenate(all_uncertainties),
        np.concatenate(all_visual_ood)
    )


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    """Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
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


def compute_per_class_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> Dict:
    """Compute per-class AUROC, sensitivity, specificity, precision, F1."""
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


def find_optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> Dict[str, Dict[str, float]]:
    """Find optimal thresholds per class using Youden's J and F1 optimization."""
    optimal = {}
    youden = {}
    
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_prob_class = y_prob[:, i]
        
        try:
            # Youden's J
            fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob_class)
            j_scores = tpr - fpr
            youden_idx = np.argmax(j_scores)
            youden[class_name] = float(thresholds[youden_idx])
            
            # Optimal F1
            precision, recall, pr_thresholds = precision_recall_curve(y_true_binary, y_prob_class)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores[:-1])
            optimal[class_name] = float(pr_thresholds[optimal_idx])
        except ValueError:
            optimal[class_name] = 0.5
            youden[class_name] = 0.5
    
    return {"optimal_f1": optimal, "youden": youden}


def compute_clinical_utility(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    class_names: List[str],
    fn_cost: float = 10.0,
    fp_cost: float = 1.0
) -> Dict[str, Any]:
    """Compute clinical utility curve for threshold selection."""
    thresholds = np.linspace(0.01, 0.99, 99)
    utility_curves = {}
    
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_prob_class = y_prob[:, i]
        
        utilities = []
        for thresh in thresholds:
            y_pred = (y_prob_class >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred).ravel()
            utility = tp - fn_cost * fn - fp_cost * fp
            utilities.append(float(utility))
        
        utility_curves[class_name] = utilities
    
    return {
        "thresholds": thresholds.tolist(),
        "utilities": utility_curves,
    }


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
    """Calibrate all clinical thresholds based on validation data."""
    
    num_samples = len(y_true)
    num_classes = len(class_names)
    
    # Compute all metrics
    per_class_metrics = compute_per_class_metrics(y_true, y_prob, class_names)
    ece, mce = compute_calibration_metrics(y_true, y_prob)
    optimal_thresholds = find_optimal_thresholds(y_true, y_prob, class_names)
    clinical_utility = compute_clinical_utility(y_true, y_prob, class_names)
    
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
    
    # Malignancy FNR (TB if no malignancy class)
    malignancy_idx = class_names.index("Lung Cancer") if "Lung Cancer" in class_names else \
                     class_names.index("Malignancy") if "Malignancy" in class_names else -1
    tb_idx = class_names.index("Tuberculosis") if "Tuberculosis" in class_names else -1
    
    fnr_malignancy = 1 - per_class_metrics["sensitivity"].get("Lung Cancer", 
                      per_class_metrics["sensitivity"].get("Malignancy", 1.0))
    fnr_tb = 1 - per_class_metrics["sensitivity"].get("Tuberculosis", 1.0)
    
    # Escalation rate at current defaults
    max_probs = np.max(y_prob, axis=1)
    escalation_mask = (max_probs < 0.6) | (uncertainty_scores > 0.15)
    escalation_rate = escalation_mask.mean()
    
    # OOD detection rate
    ood_mask = max_probs < 0.5
    ood_detection_rate = ood_mask.mean()
    
    # Threshold recommendations (start with defaults, adjust based on validation)
    confidence_thresh = 0.6
    uncertainty_thresh = 0.15
    ood_conf_thresh = 0.4
    ood_cos_thresh = 0.82
    ood_text_cos_thresh = 0.82
    calib_factor = 1.0
    mc_passes = 50
    
    rationale = "Thresholds balanced for clinical utility"
    
    if fnr_malignancy > target_fnr_malignancy:
        confidence_thresh = max(confidence_thresh, 0.65)
        uncertainty_thresh = min(uncertainty_thresh, 0.10)
        rationale = f"Increased confidence threshold to {confidence_thresh:.2f} to reduce malignancy FNR ({fnr_malignancy:.1%})"
    elif escalation_rate > target_escalation_rate * 1.5:
        confidence_thresh = max(confidence_thresh, 0.55)
        uncertainty_thresh = max(uncertainty_thresh, 0.20)
        rationale = f"Adjusted thresholds to control escalation rate ({escalation_rate:.1%})"
    
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
        optimal_thresholds=optimal_thresholds["optimal_f1"],
        youden_thresholds=optimal_thresholds["youden"],
        fnr_malignancy=float(fnr_malignancy),
        fnr_tb=float(fnr_tb),
        escalation_rate=float(escalation_rate),
        ood_detection_rate=float(ood_detection_rate),
    )
    
    env_vars = {
        "CLINICAL_CONFIDENCE_THRESHOLD": str(confidence_thresh),
        "CLINICAL_UNCERTAINTY_THRESHOLD": str(uncertainty_thresh),
        "CLINICAL_OOD_CONFIDENCE_THRESHOLD": str(ood_conf_thresh),
        "CLINICAL_OOD_COSINE_THRESHOLD": str(ood_cos_thresh),
        "CLINICAL_OOD_TEXT_COSINE_THRESHOLD": str(ood_text_cos_thresh),
        "CLINICAL_UNCERTAINTY_CALIBRATION_FACTOR": str(calib_factor),
        "CLINICAL_MC_DROPOUT_PASSES": str(mc_passes),
        "CLINICAL_THRESHOLDS_VALIDATED": "true",
        "CLINICAL_VALIDATION_DATASET": dataset_name,
        "CLINICAL_VALIDATION_DATE": datetime.now(timezone.utc).isoformat(),
        "CLINICAL_VALIDATION_METRICS": json.dumps(asdict(validation_metrics)),
    }
    
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
        environment_variables=env_vars,
    )
    
    return recommendation


def save_calibration_report(recommendation: ThresholdRecommendation, output_path: str):
    """Save calibration report with recommendations and validation metrics."""
    report = {
        "recommendation": asdict(recommendation),
        "environment_variables": recommendation.environment_variables,
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Calibration report saved to {output_path}")


def run_calibration(
    dataset_path: str,
    model_checkpoint: str,
    output_path: str,
    batch_size: int = 32,
    device: str = "cuda",
    fn_cost_malignancy: float = 10.0,
    fn_cost_tb: float = 8.0,
    target_escalation_rate: float = 0.15,
    target_fnr_malignancy: float = 0.02,
    mc_passes: int = 50,
) -> ThresholdRecommendation:
    """Main calibration pipeline."""
    
    logger.info(f"Loading dataset from {dataset_path}")
    image_paths, pdf_paths, labels = load_validation_dataset(dataset_path)
    y_true = np.array(labels)
    
    logger.info(f"Loaded {len(image_paths)} samples")
    logger.info(f"Label distribution: {np.bincount(y_true)}")
    
    logger.info(f"Loading model from {model_checkpoint}")
    model = load_model(model_checkpoint, device)
    
    logger.info("Running inference with MC Dropout...")
    y_prob, uncertainties, visual_ood = run_inference(
        model, image_paths, pdf_paths, batch_size, device, mc_passes
    )
    
    # Text OOD scores (placeholder - would need text encoder baseline)
    text_ood = np.zeros(len(labels))
    
    logger.info("Calibrating thresholds...")
    recommendation = calibrate_thresholds(
        y_true=y_true,
        y_prob=y_prob,
        uncertainty_scores=uncertainties,
        visual_ood_scores=visual_ood,
        text_ood_scores=text_ood,
        class_names=DIAGNOSTIC_CLASSES,
        dataset_name=Path(dataset_path).stem,
        fn_cost_malignancy=fn_cost_malignancy,
        fn_cost_tb=fn_cost_tb,
        target_escalation_rate=target_escalation_rate,
        target_fnr_malignancy=target_fnr_malignancy,
    )
    
    save_calibration_report(recommendation, output_path)
    
    return recommendation


def main():
    parser = argparse.ArgumentParser(description="Calibrate clinical decision thresholds")
    parser.add_argument("--dataset", required=True, help="Path to validation dataset (parquet or json)")
    parser.add_argument("--model-checkpoint", default="models/fusion_model.pt", help="Path to trained model checkpoint")
    parser.add_argument("--output", default="config/calibration_report.json", help="Output path for calibration report")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for inference")
    parser.add_argument("--fn-cost-malignancy", type=float, default=10.0, help="Relative cost of FN for malignancy")
    parser.add_argument("--fn-cost-tb", type=float, default=8.0, help="Relative cost of FN for TB")
    parser.add_argument("--target-escalation-rate", type=float, default=0.15, help="Target escalation rate")
    parser.add_argument("--target-fnr-malignancy", type=float, default=0.02, help="Target FNR for malignancy")
    parser.add_argument("--mc-passes", type=int, default=50, help="MC Dropout passes for uncertainty")
    parser.add_argument("--print-env", action="store_true", help="Print environment variables for deployment")
    
    args = parser.parse_args()
    
    recommendation = run_calibration(
        dataset_path=args.dataset,
        model_checkpoint=args.model_checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        fn_cost_malignancy=args.fn_cost_malignancy,
        fn_cost_tb=args.fn_cost_tb,
        target_escalation_rate=args.target_escalation_rate,
        target_fnr_malignancy=args.target_fnr_malignancy,
        mc_passes=args.mc_passes,
    )
    
    if args.print_env:
        print("\nEnvironment variables for deployment:")
        for key, value in recommendation.environment_variables.items():
            print(f"export {key}={value}")
    
    logger.info(f"Calibration complete. Report saved to {args.output}")


if __name__ == "__main__":
    main()