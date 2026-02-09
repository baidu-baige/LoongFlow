import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any
from sklearn.metrics import f1_score

# Task-adaptive type definitions
y = pd.DataFrame           # Target DataFrame containing 'category_id'
Predictions = np.ndarray    # Logits or class indices as numpy arrays

def ensemble(
    all_val_preds: Dict[str, Predictions],
    all_test_preds: Dict[str, Predictions],
    y_val: y,
) -> Predictions:
    """
    Combines predictions from multiple models (folds) using weighted logit averaging.
    
    This implementation leverages GPU acceleration to handle the large logit matrices 
    (approx. 115GB per model for 64,500 classes) and optimizes memory by 
    performing iterative aggregation.
    """
    print(f"Starting ensemble for {len(all_test_preds)} models...")

    # Step 1: Evaluate individual model performance on validation data
    # We use Macro F1 as per competition requirements.
    # y_val['category_id'] contains the internal encoded labels (0-64499).
    y_true = y_val['category_id'].values
    model_weights = {}
    
    for model_name, val_logits in all_val_preds.items():
        # argmax on CPU is fine for validation sets as they are 20% of training (~350k samples)
        val_preds_idx = np.argmax(val_logits, axis=1)
        score = f1_score(y_true, val_preds_idx, average='macro')
        print(f"Model {model_name} - Validation Macro F1: {score:.4f}")
        # According to specification: "Equal weights across folds (e.g., 0.2 per fold for 5-fold CV)"
        model_weights[model_name] = 1.0 / len(all_test_preds)

    # Step 2: Apply ensemble strategy on test set
    # Using torch on GPU to handle 115GB+ logit matrices efficiently.
    # We use GPU 0 for the accumulator.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    first_model = list(all_test_preds.keys())[0]
    num_test_samples, num_classes = all_test_preds[first_model].shape
    
    print(f"Aggregating test logits for {num_test_samples} samples across {num_classes} classes...")
    
    # Initialize accumulator on GPU
    # If the logit matrix (115GB) doesn't fit in GPU memory (141GB), we process in chunks.
    # Given H20-3e has 141GB, it barely fits. To be safe, we use a chunked approach.
    
    final_test_preds = np.zeros(num_test_samples, dtype=np.int32)
    chunk_size = 50000  # Process 50k images at a time to stay well within 141GB VRAM
    
    for start_idx in range(0, num_test_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, num_test_samples)
        
        # Accumulator for the current chunk
        chunk_sum_logits = torch.zeros((end_idx - start_idx, num_classes), device=device, dtype=torch.float32)
        
        for model_name, test_logits in all_test_preds.items():
            # Slice the numpy array chunk and move to GPU
            weight = model_weights[model_name]
            logits_chunk = torch.from_numpy(test_logits[start_idx:end_idx]).to(device, non_blocking=True)
            
            # Weighted average aggregation
            chunk_sum_logits.add_(logits_chunk, alpha=weight)
            
            # Explicitly clear chunk from GPU to free memory
            del logits_chunk
            
        # Get predictions for the chunk
        chunk_preds = torch.argmax(chunk_sum_logits, dim=1).cpu().numpy()
        final_test_preds[start_idx:end_idx] = chunk_preds
        
        # Clear accumulator from GPU
        del chunk_sum_logits
        torch.cuda.empty_cache()
        
        print(f"Processed test chunk {start_idx} to {end_idx}...")

    # Verification
    if np.isnan(final_test_preds).any():
        raise ValueError("Ensemble produced NaN values in predictions.")
        
    print("Ensemble completion successful.")
    return final_test_preds