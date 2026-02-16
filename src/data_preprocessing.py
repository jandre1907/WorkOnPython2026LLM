"""
Data Preprocessing Module

This module handles text tokenization and data loading for the LLM.
For simplicity, we use character-level tokenization with NumPy.
"""

import numpy as np
import os

class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, text=None):
        if text:
            self.vocab = sorted(set(text))
        else:
            # Basic ASCII
            self.vocab = [chr(i) for i in range(32, 127)]  # printable ASCII
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, text):
        """Encode text to indices."""
        return np.array([self.char_to_idx[ch] for ch in text if ch in self.char_to_idx], dtype=np.int32)

    def decode(self, indices):
        """Decode indices to text."""
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        return ''.join(self.idx_to_char[int(idx)] for idx in indices if int(idx) in self.idx_to_char)


class TextDataset:
    """Dataset for text sequences."""

    def __init__(self, data, seq_length):
        """Initialize dataset.
        
        Args:
            data: Encoded text as NumPy array
            seq_length: Length of sequences
        """
        self.data = data
        self.seq_length = seq_length
        self.num_samples = len(self.data) - self.seq_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Get a sample (x, y) pair."""
        x = self.data[idx:idx + self.seq_length].copy()
        y = self.data[idx + 1:idx + self.seq_length + 1].copy()
        return x, y


class DataLoader:
    """Simple NumPy-based data loader with batching."""
    
    def __init__(self, dataset, batch_size=32, shuffle=True, seed=42):
        """Initialize data loader.
        
        Args:
            dataset: TextDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            seed: Random seed
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        self.indices = np.arange(len(dataset))
        
    def __iter__(self):
        """Iterate over batches."""
        if self.shuffle:
            self.rng.shuffle(self.indices)
        
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            
            # Get samples
            batch_x = []
            batch_y = []
            for idx in batch_indices:
                x, y = self.dataset[idx]
                batch_x.append(x)
                batch_y.append(y)
            
            # Stack into batch
            batch_x = np.stack(batch_x)  # (batch_size, seq_length)
            batch_y = np.stack(batch_y)  # (batch_size, seq_length)
            
            yield batch_x, batch_y
    
    def __len__(self):
        """Number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def load_sample_data(seq_length=128, batch_size=32):
    """Load sample data from a small text file for demonstration.
    
    Args:
        seq_length: Length of sequences
        batch_size: Batch size
    
    Returns:
        (data_loader, tokenizer)
    """
    # Create or load sample text
    sample_text = """The quick brown fox jumps over the lazy dog. 
    This is a simple example of text data for training a language model.
    In a real scenario, you would load a large corpus of text.
    The transformer model can learn patterns from this text and generate new sequences.
    Character-level tokenization is simple but effective for small vocabularies.
    For larger models, you would typically use subword tokenization like BPE or WordPiece.
    This demonstrates how to build an LLM from scratch using only NumPy.
    """ * 20  # Repeat to get more data
    
    # Create tokenizer
    tokenizer = CharTokenizer(sample_text)
    
    # Encode text
    data = tokenizer.encode(sample_text)
    
    # Create dataset
    dataset = TextDataset(data, seq_length)
    
    # Create dataloader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader, tokenizer