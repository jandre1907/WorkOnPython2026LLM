"""
Main Module

Entry point for the LLM project using NumPy.
"""

import os
import numpy as np
from .data_preprocessing import load_sample_data
from .model import Transformer, create_mask
from .training import train_model
from .evaluation import calculate_perplexity, generate_text

# Set thermal guard API endpoint if not already set
if 'THERMAL_API_URL' not in os.environ:
    os.environ['THERMAL_API_URL'] = 'http://172.29.128.1:8085/data.json'

def main():
    """Main training and evaluation pipeline."""
    
    print("="*70)
    print("LLM from Scratch - NumPy Implementation")
    print("="*70)
    print()
    
    # Hyperparameters
    vocab_size = 95  # For char-level
    embed_size = 128  # Reduced for faster training on CPU
    num_layers = 2   # Reduced layers for CPU
    heads = 4        # Reduced heads
    forward_expansion = 2
    dropout = 0.1
    max_length = 128
    num_epochs = 3   # Reduced for demo
    batch_size = 8   # Small batch for CPU
    learning_rate = 0.001
    
    print("Hyperparameters:")
    print(f"  Vocab Size: {vocab_size}")
    print(f"  Embedding Size: {embed_size}")
    print(f"  Number of Layers: {num_layers}")
    print(f"  Number of Heads: {heads}")
    print(f"  Max Sequence Length: {max_length}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Number of Epochs: {num_epochs}")
    print()
    
    # Load data
    print("Loading data...")
    train_loader, tokenizer = load_sample_data(seq_length=max_length, batch_size=batch_size)
    print(f"  Data loaded. Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"  Number of training batches: {len(train_loader)}")
    print()
    
    # Create model
    print("Creating model...")
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        embed_size=embed_size,
        num_layers=num_layers,
        heads=heads,
        forward_expansion=forward_expansion,
        dropout=dropout,
        max_length=max_length,
        seed=42
    )
    print("  Model created successfully!")
    print()
    
    # Train
    print("Starting training...")
    train_model(model, train_loader, num_epochs, 'models/', learning_rate)
    print("  Training completed!")
    print()
    
    # Evaluate
    print("Evaluating model...")
    perplexity = calculate_perplexity(model, train_loader, max_length)
    print(f"  Perplexity: {perplexity:.4f}")
    print()
    
    # Generate sample text
    print("Generating text...")
    generated_text = generate_text(model, tokenizer, "The ", 30, temperature=0.7)
    print(f"  Generated text: {generated_text}")
    print()
    
    print("="*70)
    print("Training completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()