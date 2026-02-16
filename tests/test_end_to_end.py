import os
import numpy as np
from src.data_preprocessing import CharTokenizer


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def test_train_then_generate_simple_model():
    # Prepare tiny corpus
    text = ("hello world " * 8)
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)

    # Build (x,y) pairs where x is char at t and y is char at t+1
    xs = []
    ys = []
    for i in range(len(data) - 1):
        xs.append(data[i])
        ys.append(data[i + 1])
    X = np.array(xs, dtype=np.int32)  # (N,)
    Y = np.array(ys, dtype=np.int32)

    vocab_size = tokenizer.vocab_size
    embed = 32

    # Initialize tiny model parameters
    rng = np.random.RandomState(0)
    E = rng.normal(0, 0.1, (vocab_size, embed))
    W_out = rng.normal(0, 0.1, (embed, vocab_size))
    b = np.zeros((vocab_size,))

    def forward_idx(idx_array):
        # idx_array: (batch,)
        h = E[idx_array]  # (batch, embed)
        logits = h.dot(W_out) + b  # (batch, vocab)
        return logits, h

    # Training: simple SGD on whole dataset for next-char prediction
    lr = 1.0
    epochs = 200
    batch = len(X)
    for epoch in range(epochs):
        logits, h = forward_idx(X)
        ex = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = ex / np.sum(ex, axis=1, keepdims=True)

        # compute cross-entropy gradient on logits
        grads = probs
        grads[np.arange(batch), Y] -= 1
        grads /= batch

        # grad W_out = h^T @ grads
        grad_W = h.T.dot(grads)
        grad_b = grads.sum(axis=0)

        W_out -= lr * grad_W
        b -= lr * grad_b

        # Simple learning rate decay
        lr *= 0.9995

    # Generation (temperature 0 => argmax) using last-token autoregressive rule
    seed = os.environ.get('GENERATE_SEED', 'hello ')
    gen_len = 9
    seq = list(tokenizer.encode(seed))
    for _ in range(gen_len):
        last = np.array([seq[-1]], dtype=np.int32)
        logits, _ = forward_idx(last)
        next_tok = int(np.argmax(logits[0]))
        seq.append(next_tok)

    out = tokenizer.decode(np.array(seq))

    # Expect the generated continuation to be similar to 'hello world'
    expected = (seed + "world").strip()
    # allow some flexibility: require reasonable sequence similarity
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, expected, out[: len(expected)]).ratio()
    assert ratio >= 0.6, f"generated text not similar enough to '{expected}': {out} (ratio={ratio:.2f})"
