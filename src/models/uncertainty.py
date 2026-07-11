import torch
import numpy as np

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
        
        # Load empirical baseline standard deviations to capture non-isotropic manifold variance
        empirical_std = None
        baseline_mean = None
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
                    # Compute baseline centroid for visual epistemic uncertainty scaling
                    mean_val = np.mean(baseline_arr, axis=0)
                    baseline_mean = torch.tensor(mean_val, dtype=torch.float32, device=vision_emb.device)
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
                    perturbed_v = torch.nn.functional.normalize(perturbed_v, p=2, dim=-1)
                    
                    perturbed_t = text_emb + torch.randn_like(text_emb) * 0.05
                    perturbed_t = torch.nn.functional.normalize(perturbed_t, p=2, dim=-1)
                    
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
        
        # Calculate visual OOD distance (1 - cosine similarity) to baseline centroid
        # to detect when the frozen visual backbone has mapped a completely OOD scan
        # to a stable but incorrect coordinate in the latent space.
        visual_ood_distance = torch.zeros(batch_size, device=vision_emb.device)
        if baseline_mean is not None:
            norm_bm = baseline_mean.norm()
            norm_ve = vision_emb.norm(dim=-1)
            if norm_bm > 0:
                # Batch cosine similarity
                cos_sim = (vision_emb @ baseline_mean) / (torch.clamp(norm_ve, min=1e-8) * norm_bm)
                visual_ood_distance = torch.clamp(1.0 - cos_sim, min=0.0)

        # Compute visual uncertainty score from TTA std if available
        if visual_std is not None:
            if isinstance(visual_std, list):
                visual_std_t = torch.tensor(visual_std, dtype=torch.float32, device=vision_emb.device)
            else:
                visual_std_t = visual_std.to(vision_emb.device)
            visual_uncertainty = visual_std_t.norm(dim=-1) / (visual_std_t.shape[-1] ** 0.5)
        else:
            visual_uncertainty = torch.full((batch_size,), float('nan'), device=vision_emb.device)
            
        # Note: This is an empirical heuristic combining classification variance (from head dropout)
        # and manifold OOD distance (from input cosine similarity). It is not a closed-form derivation
        # of the Law of Total Variance but serves as a practical uncertainty score for out-of-distribution inputs.
        # Dynamically scale by prediction confidence (conf) to calibrate against temperature.
        import os
        beta = conf * float(os.getenv("UNCERTAINTY_OOD_SCALE", "0.5"))
        manifold_var = (visual_ood_distance * beta) ** 2
        combined = torch.sqrt(var_Y + manifold_var)
        
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
