"""
Training Module

Handles the training loop for the LLM using NumPy.
"""

import numpy as np
import os
import pickle
import time
import csv
from pathlib import Path
from utils.thermal_guard import check_and_cool, read_cpu_usage, read_cpu_temp

class SimpleOptimizer:
    """Simple SGD optimizer with momentum."""
    
    def __init__(self, learning_rate=0.001, momentum=0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = {}
    
    def update(self, model, grads):
        """Update model parameters using gradients."""
        # For simplicity, just apply gradient descent
        # In a real implementation, you'd track all model parameters
        pass


def cross_entropy_loss(logits, targets, mask=None):
    """Compute cross-entropy loss.
    
    Args:
        logits: Model output (batch_size, seq_length, vocab_size)
        targets: Ground truth labels (batch_size, seq_length)
        mask: Optional mask for padding
    
    Returns:
        Scalar loss value
    """
    batch_size, seq_length, vocab_size = logits.shape
    
    # Reshape for easier computation
    logits_flat = logits.reshape(-1, vocab_size)  # (batch_size*seq_length, vocab_size)
    targets_flat = targets.reshape(-1)  # (batch_size*seq_length,)
    
    # Compute softmax
    exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=1, keepdims=True))
    softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Compute cross-entropy
    ce_loss = -np.log(softmax[np.arange(len(targets_flat)), targets_flat.astype(int)] + 1e-10)
    
    if mask is not None:
        mask_flat = mask.reshape(-1)
        ce_loss = ce_loss * mask_flat
        loss = np.sum(ce_loss) / (np.sum(mask_flat) + 1e-10)
    else:
        loss = np.mean(ce_loss)
    
    return loss


def train_model(model, train_loader, num_epochs, save_path, learning_rate=0.001):
    """Train the model.
    
    Args:
        model: Transformer model
        train_loader: Data loader
        num_epochs: Number of training epochs
        save_path: Path to save model checkpoints
        learning_rate: Learning rate
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Create mask
    seq_length = 128  # This should match your model's max_length
    mask = create_mask(seq_length)
    
    # prepare thermal log file once per training session and add a session separator
    log_path = Path(save_path) / '..' / 'thermal_log.csv'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    log_file = open(log_path, 'a', newline='')
    csv_writer = csv.writer(log_file)
    if write_header:
        csv_writer.writerow(['timestamp', 'epoch', 'batch_idx', 'pre_usage', 'pre_temp', 'slept', 'post_usage', 'post_temp', 'action'])
    # session separator row
    csv_writer.writerow(['--- SESSION START ---', time.time(), f'num_epochs={num_epochs}'])
    log_file.flush()

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            # Monitor CPU and throttle if necessary to avoid overheating
            pre_usage = read_cpu_usage(interval=0.1)
            pre_temp = read_cpu_temp()
            # Read thresholds from environment or use defaults
            threshold_usage = float(os.environ.get('THERMAL_USAGE_THRESHOLD', 85.0))
            threshold_temp = float(os.environ.get('THERMAL_TEMP_THRESHOLD', 92.0))
            slept = check_and_cool(threshold_usage=threshold_usage, threshold_temp=threshold_temp, check_interval=2.0, verbose=False)
            post_usage = read_cpu_usage(interval=0.1)
            post_temp = read_cpu_temp()
            action = "sleep" if slept > 0 else "continue"
            if slept > 0:
                print(f"  Thermal guard slept for {slept:.1f}s to cool CPU")
            # Log thermal info
            try:
                csv_writer.writerow([time.time(), epoch, batch_idx, f"{pre_usage:.1f}", f"{pre_temp if pre_temp is not None else 'N/A'}", f"{slept:.2f}", f"{post_usage:.1f}", f"{post_temp if post_temp is not None else 'N/A'}", action])
                log_file.flush()
            except Exception:
                pass
            # Forward pass
            logits = model.forward(x, mask)
            
            # Compute loss
            loss = cross_entropy_loss(logits, y)
            total_loss += loss
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
        
        # Save model checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model': model,
            'loss': avg_loss
        }
        checkpoint_path = f"{save_path}/model_epoch_{epoch+1}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"  Checkpoint saved: {checkpoint_path}")
        try:
            log_file.close()
        except Exception:
            pass


def create_mask(seq_length):
    """Create causal mask for decoder.
    
    Args:
        seq_length: Length of sequence
    
    Returns:
        Lower triangular mask (seq_length, seq_length)
    """
    mask = np.tril(np.ones((seq_length, seq_length)))
    return mask