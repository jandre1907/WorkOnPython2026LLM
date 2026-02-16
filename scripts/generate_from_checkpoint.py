"""
Generate text from a saved checkpoint.

Usage example:
  python scripts/generate_from_checkpoint.py --ckpt models/model_epoch_3.pkl --prompt "Hello" --length 100

Notes:
- Checkpoints are pickled Python objects. Only load checkpoints you trust.
- This script prefers a `tokenizer` object saved inside the checkpoint under the key 'tokenizer'.
  If not present it will fall back to the library's `CharTokenizer`.
"""

import argparse
import pickle
import sys
import os
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path so `from src...` imports work when running the script
proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_root))

from src.data_preprocessing import CharTokenizer


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def sample_from_logits(logits, temperature=1.0, top_k=None):
    logits = logits.astype(np.float64)
    if temperature != 1.0 and temperature > 0:
        logits = logits / float(temperature)
    if top_k is not None and top_k > 0:
        # zero-out everything except top_k
        top_indices = np.argsort(logits)[-top_k:]
        mask = np.ones_like(logits, dtype=bool)
        mask[top_indices] = False
        logits[mask] = -1e10
    probs = softmax(logits)
    # sample
    return np.random.choice(len(probs), p=probs)


def generate(model, tokenizer, prompt, gen_len=100, temperature=1.0, top_k=None):
    max_len = getattr(model, 'max_length', 128)
    seq = tokenizer.encode(prompt)
    seq = list(map(int, seq.tolist())) if hasattr(seq, 'tolist') else list(map(int, seq))

    for _ in range(gen_len):
        # Prepare input of shape (1, seq_len)
        x = np.zeros((1, max_len), dtype=np.int32)
        # take last max_len tokens
        window = seq[-max_len:]
        x[0, max_len - len(window):] = window
        mask = None
        try:
            logits = model.forward(x, mask)
        except Exception as e:
            print("Model forward failed:", e)
            break
        # logits shape (1, seq_len, vocab)
        last_logits = logits[0, -1]  # last token logits
        next_token = sample_from_logits(last_logits, temperature=temperature, top_k=top_k)
        seq.append(int(next_token))

    return tokenizer.decode(np.array(seq))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint pickle')
    parser.add_argument('--prompt', default='', help='Prompt text')
    parser.add_argument('--length', type=int, default=128, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=0, help='Top-k filtering (0 = disabled)')
    args = parser.parse_args()

    # Load checkpoint (pickle) — only load trusted files
    with open(args.ckpt, 'rb') as f:
        ckpt = pickle.load(f)

    # Prefer tokenizer saved inside checkpoint
    tokenizer = ckpt.get('tokenizer') if isinstance(ckpt, dict) else None
    model = ckpt.get('model') if isinstance(ckpt, dict) else None
    if tokenizer is None:
        # If checkpoint doesn't include a tokenizer, try to infer a safe tokenizer
        # whose vocab size matches the model embedding to avoid index errors.
        vocab_size = None
        if model is not None and hasattr(model, 'word_embedding'):
            try:
                vocab_size = int(getattr(model, 'word_embedding').shape[0])
            except Exception:
                vocab_size = None
        if vocab_size is not None and vocab_size > 0 and vocab_size <= 95:
            # Use the first `vocab_size` printable ASCII characters as a fallback vocab
            fallback_text = ''.join(chr(i) for i in range(32, 32 + vocab_size))
            tokenizer = CharTokenizer(fallback_text)
        else:
            tokenizer = CharTokenizer()
    if model is None:
        print('Checkpoint does not contain a `model` object.', file=sys.stderr)
        sys.exit(2)

    out = generate(model, tokenizer, args.prompt, gen_len=args.length, temperature=args.temperature, top_k=(args.top_k or None))
    print(out)


if __name__ == '__main__':
    main()
