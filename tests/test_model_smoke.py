import numpy as np
from src.model import Transformer


def test_transformer_forward_shape():
    vocab_size = 16
    embed_size = 32
    num_layers = 1
    heads = 4
    max_length = 64

    model = Transformer(vocab_size=vocab_size, embed_size=embed_size, num_layers=num_layers, heads=heads, max_length=max_length)

    batch = 2
    seq_length = 10
    x = np.random.randint(0, vocab_size, size=(batch, seq_length), dtype=np.int32)

    logits = model.forward(x)

    assert logits.shape == (batch, seq_length, vocab_size)
    assert np.isfinite(logits).all()
