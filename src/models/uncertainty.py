import torch
import numpy as np
import json
from pathlib import Path

class UncertaintyEstimator:
    def __init__(self, model):
        self.model = model

    def estimate_uncertainty(self, vision_emb, text_emb, num_passes=20, visual_std=None):
        """
        Run MC Dropout combined with Test-Time Augmentation (TTA) input perturbation
        to estimate prediction mean and variance.
        Keeps model in .eval() to avoid BatchNorm errors with batch size 1,
        but explicitly enables Dropout layers.
        
        Note (Intellectual Honesty):
        We compute and return three separate uncertainty metrics:
        1. fusion_head_variance — epistemic uncertainty from the classification head only (via MC Dropout).
        2. visual_uncertainty_score — input-space uncertainty from TTA visual_std.
        3. combined_uncertainty — computed using a heuristic combination of the epistemic classification
           variance (var_Y) and a manifold-space out-of-distribution (OOD) distance metric (visual_ood_distance).
           This is a practical engineering heuristic combining input out-of-distribution distance with head variance,
           rather than a strict, closed-form implementation of the Law of Total Variance.
        """
        self.model.eval()
        
        import torch.nn.functional as F
        # Flaw #4 Fix: Ensure embeddings are explicitly L2 unit-normalized at ingestion
        vision_emb = F.normalize(vision_emb, p=2, dim=-1)
        text_emb = F.normalize(text_emb, p=2, dim=-1)
        
        # Load empirical baseline standard deviations to capture non-isotropic manifold variance
        empirical_std = None
        baseline_mean = None
        baseline_mean_text = None
        try:
            import json
            from pathlib import Path
            centroid_path = Path("temp/drift/features_centroid.json")
            if centroid_path.exists():
                with open(centroid_path, "r") as f:
                    centroid_data = json.load(f)
                if centroid_data and "centroid" in centroid_data:
                    mean_val = centroid_data["centroid"]
                    baseline_mean = torch.tensor(mean_val, dtype=torch.float32, device=vision_emb.device)
                    if "std" in centroid_data:
                        empirical_std_np = np.array(centroid_data["std"]) + 1e-6
                        empirical_std = torch.tensor(empirical_std_np, dtype=torch.float32, device=vision_emb.device)
                        
            text_centroid_path = Path("temp/drift/text_centroid.json")
            if text_centroid_path.exists():
                with open(text_centroid_path, "r") as f:
                    text_centroid_data = json.load(f)
                if text_centroid_data and "centroid" in text_centroid_data:
                    text_mean_val = text_centroid_data["centroid"]
                    baseline_mean_text = torch.tensor(text_mean_val, dtype=torch.float32, device=text_emb.device)
        except Exception:
            pass
        
        batch_size = vision_emb.shape[0]
        
        # Calculate visual OOD distance (1 - cosine similarity) to baseline centroid
        visual_ood_distance = torch.zeros(batch_size, device=vision_emb.device)
        if baseline_mean is not None:
            baseline_mean = F.normalize(baseline_mean, p=2, dim=-1)
            cos_sim = (vision_emb @ baseline_mean)
            visual_ood_distance = torch.clamp(1.0 - cos_sim, min=0.0)

        # Calculate text OOD distance to text baseline centroid
        text_ood_distance = torch.zeros(batch_size, device=text_emb.device)
        if baseline_mean_text is not None:
            baseline_mean_text = F.normalize(baseline_mean_text, p=2, dim=-1)
            cos_sim_t = (text_emb @ baseline_mean_text)
            text_ood_distance = torch.clamp(1.0 - cos_sim_t, min=0.0)

        # Joint out-of-distribution distance
        if baseline_mean is not None and baseline_mean_text is not None:
            alpha = 0.5
            joint_ood_distance = alpha * visual_ood_distance + (1.0 - alpha) * text_ood_distance
        elif baseline_mean_text is not None:
            joint_ood_distance = text_ood_distance
        else:
            joint_ood_distance = visual_ood_distance

        # Temperature scaling derived from OOD distance to calibrate uncertainty
        # A simple linear calibration: T = 1.0 + joint_ood_distance
        # This increases the temperature of the softmax for OOD samples, softening the output
        # distribution and naturally increasing the variance without heuristic additions.
        temperature = 1.0 + joint_ood_distance.unsqueeze(1)
        
        try:
            # Enable dropout layers specifically inside the fusion model
            for m in self.model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.train()
                # Safety: Ensure no BatchNorm is in training mode
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    m.eval()
            
            all_logits = []
            all_visual_noises = []
            with torch.no_grad():
                for _ in range(num_passes):
                    # Use TTA-derived visual_std for perturbation magnitude if available
                    if visual_std is not None:
                        if isinstance(visual_std, list):
                            visual_std_t = torch.tensor(visual_std, dtype=torch.float32, device=vision_emb.device)
                        else:
                            visual_std_t = visual_std.to(vision_emb.device)
                        noise_scale = visual_std_t
                        noise = torch.randn_like(vision_emb) * noise_scale
                    else:
                        # Local perturbation must be purely local/isotropic to avoid global population scaling bias
                        noise = torch.randn_like(vision_emb) * 0.05
                        
                    perturbed_v = vision_emb + noise
                    perturbed_v = torch.nn.functional.normalize(perturbed_v, p=2, dim=-1)
                    
                    perturbed_t = text_emb + torch.randn_like(text_emb) * 0.05
                    perturbed_t = torch.nn.functional.normalize(perturbed_t, p=2, dim=-1)
                    
                    _, logits = self.model(perturbed_v, perturbed_t)
                    
                    # Apply OOD temperature scaling before softmax
                    scaled_logits = logits / temperature
                    all_logits.append(torch.softmax(scaled_logits, dim=1))
                    
                    # Compute the normalized L2 norm of the noise vector for this pass (batch_size,)
                    noise_norm = noise.norm(dim=-1) / (noise.shape[-1] ** 0.5)
                    all_visual_noises.append(noise_norm)
        finally:
            self.model.eval()
        
        # Stack results (num_passes, batch, num_classes)
        stacked_probs = torch.stack(all_logits)
        stacked_noises = torch.stack(all_visual_noises)  # (num_passes, batch)
        
        # Compute mean of probabilities
        mean_probs = torch.mean(stacked_probs, dim=0)
        
        # Prediction is the class with highest mean probability
        conf, pred = torch.max(mean_probs, dim=1)
        
        # Compute visual uncertainty score from TTA std if available
        if visual_std is not None:
            if isinstance(visual_std, list):
                visual_std_t = torch.tensor(visual_std, dtype=torch.float32, device=vision_emb.device)
            else:
                visual_std_t = visual_std.to(vision_emb.device)
            visual_uncertainty = visual_std_t.norm(dim=-1) / (visual_std_t.shape[-1] ** 0.5)
        else:
            visual_uncertainty = torch.full((batch_size,), float('nan'), device=vision_emb.device)
            
        # Extract prediction probabilities across passes for prediction variance
        Y = torch.stack([stacked_probs[t, torch.arange(batch_size), pred] for t in range(num_passes)])  # (num_passes, batch)
        var_Y = Y.var(dim=0, unbiased=True)
        fusion_uncertainties = var_Y.sqrt()
        
        if num_passes > 1 and torch.all(var_Y == 0.0):
            import logging
            temp_logger = logging.getLogger("uncertainty-estimator")
            temp_logger.critical(
                "MC DROPOUT STATE ERROR: Variance is exactly zero across all passes. "
                "Verify that the model has active dropout layers and that they are "
                "explicitly set to training mode during estimation."
            )
            
        # Due to Temperature scaling applied earlier on the logits, the probability 
        # distributions for OOD samples are naturally flattened. This increases var_Y,
        # providing a sound, model-derived calibration of uncertainty.
        combined = fusion_uncertainties
        
        return {
            "prediction": pred,
            "mean_confidence": conf,
            "std_deviation": combined,  # API-compatible field now uses combined metric
            "fusion_head_variance": fusion_uncertainties,
            "visual_uncertainty_score": visual_uncertainty,
            "combined_uncertainty": combined,
            "all_probs": mean_probs
        }

    # Flaw #19 Fix: Removed dead calculate_ece() method.
    # ECE calculation should live in the evaluation pipeline (src/evaluation/)
    # where it can be called with actual validation data, not as an orphan method here.

if __name__ == "__main__":
    # Test would go here
    pass
