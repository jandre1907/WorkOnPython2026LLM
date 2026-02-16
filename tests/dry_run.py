"""Tiny dry-run to exercise the training loop and thermal guard.

Creates a DummyModel that returns random logits and a small in-memory
dataset. Runs one epoch and writes a checkpoint under tests/checkpoints.
"""
import os
import sys
import numpy as np

# ensure src is importable
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('src'))
import training


class DummyModel:
    def __init__(self, vocab_size=32, seq_length=128):
        self.vocab_size = vocab_size
        self.seq_length = seq_length

    def forward(self, x, mask=None):
        # x: (batch, seq)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        # return random logits
        return np.random.randn(batch_size, seq_len, self.vocab_size).astype(np.float32)


def make_loader(num_batches=20, batch_size=1, seq_length=128, vocab_size=32):
    loader = []
    for _ in range(num_batches):
        x = np.random.randint(0, vocab_size, size=(batch_size, seq_length)).astype(np.int64)
        y = np.random.randint(0, vocab_size, size=(batch_size, seq_length)).astype(np.int64)
        loader.append((x, y))
    return loader


def main():
    save_path = 'tests/checkpoints'
    # clean checkpoints
    if os.path.exists(save_path):
        for f in os.listdir(save_path):
            try:
                os.remove(os.path.join(save_path, f))
            except Exception:
                pass
    else:
        os.makedirs(save_path, exist_ok=True)

    model = DummyModel(vocab_size=32, seq_length=128)
    loader = make_loader(num_batches=20, batch_size=1, seq_length=128, vocab_size=32)

    print('Starting dry-run training (2 epochs, 20 batches each)...')
    training.train_model(model, loader, num_epochs=2, save_path=save_path, learning_rate=0.01)
    print('Dry-run completed.')


if __name__ == '__main__':
    main()
