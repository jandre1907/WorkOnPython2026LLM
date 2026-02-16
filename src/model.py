"""
Model Module

Implements a simple Transformer-based language model using NumPy.
"""

import numpy as np
import math

class SelfAttention:
    """Self-attention mechanism."""

    def __init__(self, embed_size, heads, seed=42):
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.rng = np.random.RandomState(seed)
        
        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"
        
        # Initialize weight matrices
        scale = 1.0 / np.sqrt(self.head_dim)
        self.W_q = self.rng.normal(0, scale, (embed_size, embed_size))
        self.W_k = self.rng.normal(0, scale, (embed_size, embed_size))
        self.W_v = self.rng.normal(0, scale, (embed_size, embed_size))
        self.W_out = self.rng.normal(0, scale, (embed_size, embed_size))
        
        # Cache for backprop
        self.cache = {}
    
    def forward(self, values, keys, queries, mask=None):
        """Forward pass for self-attention."""
        batch_size = queries.shape[0]
        seq_len = queries.shape[1]
        
        # Linear projections
        Q = np.dot(queries, self.W_q)  # (batch, seq_len, embed_size)
        K = np.dot(keys, self.W_k)
        V = np.dot(values, self.W_v)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Compute attention scores
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)  # (batch, heads, seq_len, seq_len)
        
        # Apply mask if provided. Ensure mask broadcasts to (batch, heads, seq_len, seq_len).
        if mask is not None:
            try:
                if mask.ndim == 2:
                    mask_b = mask[None, None, :, :]
                else:
                    mask_b = mask
                scores = np.where(mask_b == 0, -1e10, scores)
            except Exception:
                # Fallback: ignore mask if shapes incompatible
                pass
        
        # Apply softmax
        attention_weights = self._softmax(scores, axis=-1)
        
        # Apply attention to values
        context = np.matmul(attention_weights, V)  # (batch, heads, seq_len, head_dim)
        
        # Reshape back
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_size)
        
        # Final linear projection
        output = np.dot(context, self.W_out)
        
        # Cache for backprop
        self.cache = {'Q': Q, 'K': K, 'V': V, 'attention': attention_weights, 'scores': scores}
        
        return output
    
    @staticmethod
    def _softmax(x, axis=-1):
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)


class TransformerBlock:
    """Transformer block with self-attention and feed-forward."""
    
    def __init__(self, embed_size, heads, dropout=0.1, forward_expansion=4, seed=42):
        self.embed_size = embed_size
        self.dropout = dropout
        self.rng = np.random.RandomState(seed)
        
        self.attention = SelfAttention(embed_size, heads, seed=seed)
        
        # Layer normalization parameters
        self.norm1_gamma = np.ones(embed_size)
        self.norm1_beta = np.zeros(embed_size)
        self.norm2_gamma = np.ones(embed_size)
        self.norm2_beta = np.zeros(embed_size)
        
        # Feed-forward network
        scale = 1.0 / np.sqrt(embed_size)
        self.W1 = self.rng.normal(0, scale, (embed_size, forward_expansion * embed_size))
        self.b1 = np.zeros(forward_expansion * embed_size)
        self.W2 = self.rng.normal(0, scale, (forward_expansion * embed_size, embed_size))
        self.b2 = np.zeros(embed_size)
    
    def _layer_norm(self, x, gamma, beta, eps=1e-5):
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + eps)
        return gamma * x_norm + beta
    
    def _apply_dropout(self, x):
        """Apply dropout during training."""
        if self.dropout > 0:
            mask = self.rng.binomial(1, 1 - self.dropout, x.shape) / (1 - self.dropout)
            return x * mask
        return x
    
    def forward(self, value, key, query, mask=None):
        """Forward pass through transformer block."""
        # Self-attention
        attention_output = self.attention.forward(value, key, query, mask)
        attention_output = self._apply_dropout(attention_output)
        
        # Add and normalize
        x = self._layer_norm(query + attention_output, self.norm1_gamma, self.norm1_beta)
        
        # Feed-forward network
        ff_output = np.dot(x, self.W1) + self.b1
        ff_output = np.maximum(ff_output, 0)  # ReLU
        ff_output = np.dot(ff_output, self.W2) + self.b2
        ff_output = self._apply_dropout(ff_output)
        
        # Add and normalize
        output = self._layer_norm(x + ff_output, self.norm2_gamma, self.norm2_beta)
        
        return output


class Transformer:
    """Transformer decoder for language modeling."""
    
    def __init__(self, vocab_size, embed_size, num_layers, heads, forward_expansion=4, 
                 dropout=0.1, max_length=128, seed=42):
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.rng = np.random.RandomState(seed)
        
        # Embedding matrices
        scale = 1.0 / np.sqrt(embed_size)
        self.word_embedding = self.rng.normal(0, scale, (vocab_size, embed_size))
        self.position_embedding = self.rng.normal(0, scale, (max_length, embed_size))
        
        # Transformer blocks
        self.layers = [TransformerBlock(embed_size, heads, dropout, forward_expansion, seed=seed + i)
                       for i in range(num_layers)]
        
        # Output layer
        self.W_out = self.rng.normal(0, scale, (embed_size, vocab_size))
        self.b_out = np.zeros(vocab_size)
    
    def forward(self, x, mask=None):
        """Forward pass through the model.
        
        Args:
            x: Token indices (batch_size, seq_length)
            mask: Attention mask (seq_length, seq_length)
        
        Returns:
            Logits (batch_size, seq_length, vocab_size)
        """
        batch_size, seq_length = x.shape
        
        # Embedding lookup
        embedded = self.word_embedding[x]  # (batch_size, seq_length, embed_size)
        
        # Add positional embeddings
        positions = np.arange(seq_length)
        pos_embedded = self.position_embedding[positions]
        x_embed = embedded + pos_embedded
        
        # Pass through transformer blocks
        output = x_embed
        for layer in self.layers:
            output = layer.forward(output, output, output, mask)
        
        # Project to vocabulary
        logits = np.dot(output, self.W_out) + self.b_out  # (batch_size, seq_length, vocab_size)
        
        return logits


def create_mask(seq_length):
    """Create causal mask for decoder.
    
    Args:
        seq_length: Length of the sequence
    
    Returns:
        Lower triangular mask (seq_length, seq_length)
    """
    mask = np.tril(np.ones((seq_length, seq_length)))
    return mask