import torch
import numpy as np

class UncertaintyEstimator:
    def __init__(self, model):
        self.model = model

    def estimate_uncertainty(self, vision_emb, text_emb, num_passes=20):
        """
        Run MC Dropout to estimate prediction mean and standard deviation.
        Keeps model in .eval() to avoid BatchNorm errors with batch size 1,
        but explicitly enables Dropout layers.
        
        NOTE ON LIMITATION (Tara's Critique - The MC Dropout Illusion):
        Because the visual encoder (BiomedCLIP) and text encoder (SapBERT) are run 
        exactly once during upstream pipeline execution to extract static features, 
        and because BiomedCLIP's vision tower has a dropout rate of 0.0, this function 
        only estimates epistemic uncertainty of the fusion/classification head layers 
        (LateFusionModel). Visual feature extraction uncertainty is NOT captured here. 
        To make this clear, we return both "std_deviation" (for API compatibility) 
        and "fusion_head_variance".
        """
        self.model.eval()
        
        try:
            # Enable dropout layers specifically
            for m in self.model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.train()
                # Safety: Ensure no BatchNorm is in training mode
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    m.eval()
            
            all_logits = []
            # CRITICAL CONSTRAINT (Tara's T1): We use torch.no_grad() here because MC Dropout is strictly for
            # inference-time uncertainty estimation, which benefits from no-grad memory optimization and speed.
            # However, because gradients are disabled, Grad-CAM (which requires backpropagation to compute
            # attention maps) cannot run in the same pass. Therefore, Grad-CAM and uncertainty estimation
            # must be executed in separate model passes.
            with torch.no_grad():
                for _ in range(num_passes):
                    _, logits = self.model(vision_emb, text_emb)
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
        
        # Flaw #14 Fix: Return std_deviation as a tensor instead of a Python list.
        # Consumers call .tolist() on the full results dict via the API serialization layer,
        # ensuring consistent type handling throughout the pipeline.
        batch_size = vision_emb.shape[0]
        uncertainties = torch.tensor([std_probs[i, pred[i]].item() for i in range(batch_size)])
        
        return {
            "prediction": pred,
            "mean_confidence": conf,
            "std_deviation": uncertainties,
            "fusion_head_variance": uncertainties,  # Clarified classification head variance
            "all_probs": mean_probs
        }

    # Flaw #19 Fix: Removed dead calculate_ece() method.
    # ECE calculation should live in the evaluation pipeline (src/evaluation/)
    # where it can be called with actual validation data, not as an orphan method here.

if __name__ == "__main__":
    # Test would go here
    pass
