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
        """
        self.model.eval()
        
        # Enable dropout layers specifically
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
            # Safety: Ensure no BatchNorm is in training mode
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                m.eval()
        
        all_logits = []
        with torch.no_grad():
            for _ in range(num_passes):
                _, logits = self.model(vision_emb, text_emb)
                all_logits.append(torch.softmax(logits, dim=1))
        
        # Stack results (num_passes, batch, num_classes)
        stacked_probs = torch.stack(all_logits)
        
        # Compute mean and standard deviation
        mean_probs = torch.mean(stacked_probs, dim=0)
        std_probs = torch.std(stacked_probs, dim=0)
        
        # Prediction is the class with highest mean probability
        conf, pred = torch.max(mean_probs, dim=1)
        
        # Uncertainty is the std dev of the predicted class
        batch_size = vision_emb.shape[0]
        uncertainties = [std_probs[i, pred[i]].item() for i in range(batch_size)]
        
        return {
            "prediction": pred,
            "mean_confidence": conf,
            "std_deviation": uncertainties,
            "all_probs": mean_probs
        }

    def calculate_ece(self, y_true, y_prob, n_bins=10):
        """Calculate Expected Calibration Error (ECE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = predictions == y_true
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated confidence and accuracy in each bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece

if __name__ == "__main__":
    # Test would go here
    pass
