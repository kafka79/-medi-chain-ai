import pytest
import pandas as pd
import os

def test_patient_leakage():
    """
    Verify that no patient (report_id) appears in more than one split.
    This is critical for medical AI to prevent optimistic bias.
    """
    processed_dir = "data/processed"
    train_path = os.path.join(processed_dir, 'train.csv')
    val_path = os.path.join(processed_dir, 'val.csv')
    test_path = os.path.join(processed_dir, 'test.csv')
    
    # If files don't exist, we skip (assuming pipeline hasn't run)
    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        pytest.skip("Processed data files not found. Run pipeline first.")
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    train_ids = set(train_df['report_id'].unique())
    val_ids = set(val_df['report_id'].unique())
    test_ids = set(test_df['report_id'].unique())
    
    # Check intersections
    assert train_ids.isdisjoint(val_ids), "Leakage detected between Train and Val"
    assert train_ids.isdisjoint(test_ids), "Leakage detected between Train and Test"
    assert val_ids.isdisjoint(test_ids), "Leakage detected between Val and Test"
    
    print("Success: No patient leakage detected across splits.")

if __name__ == "__main__":
    test_patient_leakage()
