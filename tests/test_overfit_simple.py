import numpy as np
from src.data_preprocessing import CharTokenizer


def cross_entropy_loss_and_probs(logits, targets):
    # logits: (batch, seq, vocab)
    # targets: (batch, seq)
    b, s, v = logits.shape
    logits_flat = logits.reshape(-1, v)
    targets_flat = targets.reshape(-1)
    # stable softmax
    ex = np.exp(logits_flat - np.max(logits_flat, axis=1, keepdims=True))
    probs = ex / np.sum(ex, axis=1, keepdims=True)
    eps = 1e-12
    loss = -np.log(probs[np.arange(len(targets_flat)), targets_flat] + eps).mean()
    probs = probs.reshape(b, s, v)
    return loss, probs


def test_simple_overfit_output_layer():
    np.random.seed(42)

    # Tiny vocabulary and dataset
    text = "ab " * 40
    tokenizer = CharTokenizer(text)
    data = tokenizer.encode(text)

    seq_length = 8
    # Build one batch (batch_size=2)
    batch_size = 2
    # Create simple sliding windows
    X = []
    Y = []
    for i in range(0, 2 * seq_length, seq_length):
        x = data[i:i + seq_length]
        y = data[i + 1:i + seq_length + 1]
        if len(x) == seq_length and len(y) == seq_length:
            X.append(x)
            Y.append(y)
    X = np.stack(X)[:batch_size]
    Y = np.stack(Y)[:batch_size]

    vocab_size = tokenizer.vocab_size
    embed = 16

    # Tiny model: embedding lookup + linear projection
    E = np.random.normal(0, 0.1, (vocab_size, embed))
    W_out = np.random.normal(0, 0.1, (embed, vocab_size))
    b = np.zeros((vocab_size,))

    def forward(x):
        # x: (batch, seq)
        hidden = E[x]  # (batch, seq, embed)
        logits = np.matmul(hidden, W_out) + b
        return logits, hidden

    # Initial loss
    logits, hidden = forward(X)
    loss0, probs = cross_entropy_loss_and_probs(logits, Y)

    lr = 1.0
    steps = 60
    prev_loss = loss0
    for step in range(steps):
        logits, hidden = forward(X)
        loss, probs = cross_entropy_loss_and_probs(logits, Y)

        # compute grad wrt logits: probs - one_hot
        bsz, slen, v = probs.shape
        grad_logits = probs.copy()
        idx = np.arange(bsz * slen)
        target_flat = Y.reshape(-1)
        grad_logits = grad_logits.reshape(-1, v)
        grad_logits[idx, target_flat] -= 1
        grad_logits = grad_logits.reshape(bsz, slen, v)
        grad_logits = grad_logits / (bsz * slen)

        # grad W_out = sum over tokens hidden^T @ grad_logits
        hid_flat = hidden.reshape(-1, embed)  # (N, embed)
        grad_logits_flat = grad_logits.reshape(-1, v)  # (N, v)
        grad_W = hid_flat.T.dot(grad_logits_flat)
        grad_b = grad_logits_flat.sum(axis=0)

        # update
        W_out -= lr * grad_W
        b -= lr * grad_b

        if step % 10 == 0:
            # observe loss
            logits, _ = forward(X)
            loss = cross_entropy_loss_and_probs(logits, Y)[0]
        # stop early if loss decreased substantially
        if loss < prev_loss - 1e-6:
            prev_loss = loss

    # final loss
    logits, _ = forward(X)
    final_loss = cross_entropy_loss_and_probs(logits, Y)[0]

    assert final_loss < loss0, f"loss did not decrease: {final_loss} >= {loss0}"
