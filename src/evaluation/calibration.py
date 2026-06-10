import numpy as np

def calculate_ece(y_true, y_prob, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE) for a set of predictions.
    y_true: 1D array/list of true class indices (e.g. 0 to K-1).
    y_prob: 2D array/list of predicted probabilities (shape: N x K).
    n_bins: number of bins to partition the confidence space [0, 1].
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Assert dimensions
    if y_prob.ndim == 1:
        # Binary case, assume y_prob represents class 1 probabilities
        y_prob = np.stack([1 - y_prob, y_prob], axis=1)
        
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Elements in bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)
