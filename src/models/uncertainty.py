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
        3. combined_uncertainty — geometric mean of (1) and (2), factoring in both
           visual feature instability AND classification head variance.
        
        This makes it explicit that fusion_head_variance alone does NOT capture
        visual encoder uncertainty (BiomedCLIP has dropout=0.0).
        """
        self.model.eval()
        
        try:
            # Enable dropout layers specifically inside the fusion model
            for m in self.model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.train()
                # Safety: Ensure no BatchNorm is in training mode
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    m.eval()
            
            all_logits = []
            with torch.no_grad():
                for _ in range(num_passes):
                    # Use TTA-derived visual_std for perturbation magnitude
                    if visual_std is not None:
                        perturbed_v = vision_emb + torch.randn_like(vision_emb) * visual_std
                    else:
                        perturbed_v = vision_emb + torch.randn_like(vision_emb) * 0.05
                        
                    perturbed_t = text_emb + torch.randn_like(text_emb) * 0.05
                    _, logits = self.model(perturbed_v, perturbed_t)
                    all_logits.append(torch.softmax(logits, dim=1))
        finally:
            self.model.eval()
        
        # Stack results (num_passes, batch, num_classes)
        stacked_probs = torch.stack(all_logits)
        
        # Compute mean and standard deviation
        mean_probs = torch.mean(stacked_probs, dim=0)
        std_probs = torch.std(stacked_probs, dim=0)
        
        # Prediction is the class with highest mean probability
        conf, pred = torch.max(mean_probs, dim=1)
        
        batch_size = vision_emb.shape[0]
        fusion_uncertainties = torch.tensor([std_probs[i, pred[i]].item() for i in range(batch_size)])
        
        # Flaw #2-structural Fix: Compute visual uncertainty score from TTA std
        # visual_std is the per-dimension std across TTA augmented images.
        # We collapse it to a scalar per sample via L2 norm / sqrt(dim).
        if visual_std is not None:
            if isinstance(visual_std, list):
                visual_std_t = torch.tensor(visual_std, dtype=torch.float32)
            else:
                visual_std_t = visual_std
            # Normalized L2 norm gives a scale-invariant instability score
            visual_uncertainty = visual_std_t.norm(dim=-1) / (visual_std_t.shape[-1] ** 0.5)
        else:
            # No TTA data available — report as unknown (NaN) rather than zero
            visual_uncertainty = torch.full((batch_size,), float('nan'))
        
        # Combined uncertainty: geometric mean of fusion-head and visual uncertainty
        # If visual_uncertainty is NaN (no TTA), combined falls back to fusion_uncertainties
        combined = torch.where(
            torch.isnan(visual_uncertainty),
            fusion_uncertainties,
            (fusion_uncertainties * visual_uncertainty).sqrt()
        )
        
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
