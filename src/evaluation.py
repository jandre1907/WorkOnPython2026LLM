"""
Evaluation Module

Evaluates the trained LLM using NumPy.
"""

import numpy as np
import math

def  calculate_perplexity(model, data_loader, seq_length=128):
    """Calculate perplexity on the dataset.
    
    Args:
        model: Transformer model
        data_loader: Data loader for evaluation
        seq_length: Sequence length
    
    Returns:
        Perplexity score
    """
    total_loss = 0
    total_tokens = 0
    
    # Create mask
    mask = create_mask(seq_length)
    
    for x, y in data_loader:
        # Forward pass
        logits = model.forward(x, mask)
        
        # Compute loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        y_flat = y.reshape(-1)
        
        # Softmax
        exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=1, keepdims=True))
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Cross-entropy
        ce_loss = -np.log(softmax[np.arange(len(y_flat)), y_flat.astype(int)] + 1e-10)
        
        total_loss += np.sum(ce_loss)
        total_tokens += len(y_flat)
    
    perplexity = math.exp(total_loss / total_tokens)
    return perplexity


def generate_text(model, tokenizer, start_text, max_length, temperature=1.0, seq_length=128):
    """Generate text using the model.
    
    Args:
        model: Transformer model
        tokenizer: CharTokenizer instance
        start_text: Starting text for generation
        max_length: Maximum length of generated text
        temperature: Temperature for sampling (higher = more random)
        seq_length: Model sequence length
    
    Returns:
        Generated text string
    """
    # Encode start text
    generated = tokenizer.encode(start_text)
    generated = np.array(generated, dtype=np.int32)
    
    # Create mask
    mask = create_mask(seq_length)
    
    rng = np.random.RandomState(42)
    
    for step in range(max_length - len(start_text)):
        # Pad sequence to seq_length if needed
        if len(generated) < seq_length:
            padding = np.zeros(seq_length - len(generated), dtype=np.int32)
            input_seq = np.concatenate([generated, padding])
        else:
            input_seq = generated[-seq_length:]
        
        # Get model prediction
        input_batch = input_seq.reshape(1, -1)
        logits = model.forward(input_batch, mask)
        
        # Get last token logits
        last_index = len(generated) - 1
        next_token_logits = logits[0, last_index, :]

        if temperature == 0:
            # Deterministic argmax sampling
            next_token = int(np.argmax(next_token_logits))
        else:
            # Apply temperature and sample
            scaled = next_token_logits / float(temperature)
            exp_logits = np.exp(scaled - np.max(scaled))
            probabilities = exp_logits / np.sum(exp_logits)
            next_token = rng.choice(len(probabilities), p=probabilities)
        generated = np.append(generated, next_token)
    
    # Decode to text
    return tokenizer.decode(generated)


def create_mask(seq_length):
    """Create causal mask for decoder.
    
    Args:
        seq_length: Length of sequence
    
    Returns:
        Lower triangular mask (seq_length, seq_length)
    """
    mask = np.tril(np.ones((seq_length, seq_length)))
    return mask