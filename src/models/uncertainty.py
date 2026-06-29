import torch
import numpy as np

class UncertaintyEstimator:
    def __init__(self, model):
        self.model = model

    def estimate_uncertainty(self, vision_emb, text_emb, num_passes=20, visual_std=None):
        """
        Run MC Dropout to estimate prediction mean and standard deviation.
        Keeps model in .eval() to avoid BatchNorm errors with batch size 1,
        but explicitly enables Dropout layers.
        
        Flaw #2-structural Fix (MC Dropout Illusion):
        We now compute and return THREE separate uncertainty metrics:
        1. fusion_head_variance — epistemic uncertainty from the classification head only.
        2. visual_uncertainty_score — input-space uncertainty from TTA visual_std.
        3. combined_uncertainty — computed via the Law of Total Variance factoring in covariance
           between the visual input noise and classification output variation.
        """
        self.model.eval()
        
        # Load empirical baseline standard deviations to capture non-isotropic manifold variance
        empirical_std = None
        try:
            import json
            from pathlib import Path
            baseline_path = Path("temp/drift/features_baseline_cache.json")
            if baseline_path.exists():
                with open(baseline_path, "r") as f:
                    baseline_features = json.load(f)
                if baseline_features and len(baseline_features) > 1:
                    baseline_arr = np.array(baseline_features)
                    empirical_std_np = np.std(baseline_arr, axis=0) + 1e-6
                    empirical_std = torch.tensor(empirical_std_np, dtype=torch.float32, device=vision_emb.device)
        except Exception:
            pass
        
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
                    # Use TTA-derived visual_std for perturbation magnitude
                    if visual_std is not None:
                        if isinstance(visual_std, list):
                            visual_std_t = torch.tensor(visual_std, dtype=torch.float32, device=vision_emb.device)
                        else:
                            visual_std_t = visual_std.to(vision_emb.device)
                        
                        if empirical_std is not None:
                            # Scale visual noise non-isotropically by relative dimension variation
                            normalized_empirical = empirical_std / (empirical_std.mean() + 1e-8)
                            noise_scale = visual_std_t * normalized_empirical
                        else:
                            noise_scale = visual_std_t
                        noise = torch.randn_like(vision_emb) * noise_scale
                    else:
                        if empirical_std is not None:
                            # Scale isotropic noise (0.05) by normalized empirical standard deviation
                            normalized_empirical = empirical_std / (empirical_std.mean() + 1e-8)
                            noise = torch.randn_like(vision_emb) * 0.05 * normalized_empirical
                        else:
                            noise = torch.randn_like(vision_emb) * 0.05
                        
                    perturbed_v = vision_emb + noise
                    perturbed_t = text_emb + torch.randn_like(text_emb) * 0.05
                    _, logits = self.model(perturbed_v, perturbed_t)
                    all_logits.append(torch.softmax(logits, dim=1))
                    
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
        
        batch_size = vision_emb.shape[0]
        
        # Extract prediction probabilities across passes for covariance
        Y = torch.stack([stacked_probs[t, torch.arange(batch_size), pred] for t in range(num_passes)])  # (num_passes, batch)
        X = stacked_noises  # (num_passes, batch)
        
        mean_Y = Y.mean(dim=0)
        mean_X = X.mean(dim=0)
        
        var_Y = Y.var(dim=0, unbiased=True)
        var_X = X.var(dim=0, unbiased=True)
        
        # Sample covariance
        cov_XY = torch.sum((X - mean_X) * (Y - mean_Y), dim=0) / (num_passes - 1)
        
        fusion_uncertainties = var_Y.sqrt()
        
        if num_passes > 1 and torch.all(var_Y == 0.0):
            import logging
            temp_logger = logging.getLogger("uncertainty-estimator")
            temp_logger.critical(
                "MC DROPOUT STATE ERROR: Variance is exactly zero across all passes. "
                "Verify that the model has active dropout layers and that they are "
                "explicitly set to training mode during estimation."
            )
        
        # Flaw #2-structural Fix: Compute visual uncertainty score from TTA std
        # visual_std is the per-dimension std across TTA augmented images.
        if visual_std is not None:
            if isinstance(visual_std, list):
                visual_std_t = torch.tensor(visual_std, dtype=torch.float32, device=vision_emb.device)
            else:
                visual_std_t = visual_std.to(vision_emb.device)
            visual_uncertainty = visual_std_t.norm(dim=-1) / (visual_std_t.shape[-1] ** 0.5)
            
            # Panel Flaw #6 Fix: Law of total variance with covariance.
            # Var(Y) = Var_fusion + Var_visual + 2 * Cov(fusion, visual)
            combined_var = var_Y + var_X + 2.0 * cov_XY
            combined = torch.sqrt(torch.clamp(combined_var, min=0.0))
        else:
            # No TTA data available — report as unknown (NaN) rather than zero
            visual_uncertainty = torch.full((batch_size,), float('nan'), device=vision_emb.device)
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
