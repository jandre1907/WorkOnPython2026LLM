"""Run a minimal end-to-end training + generation locally.

This trains a very small Transformer on a tiny corpus and prints a generated sample.
Configuration comes from (in priority): CLI args -> environment -> .env file -> defaults.
Optionally load a checkpoint instead of training.
"""

from pathlib import Path
import sys
proj_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(proj_root))

import argparse
import os
import pickle
import numpy as np
from src.data_preprocessing import CharTokenizer
from src.model import Transformer
from src.training import train_model
from src.evaluation import generate_text


def load_dotenv(path):
    env = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env


def coerce(val, kind):
    if val is None:
        return None
    try:
        return kind(val)
    except Exception:
        return val


def main():
    parser = argparse.ArgumentParser(description='Minimal E2E trainer and generator')
    parser.add_argument('--num-epochs', type=int, help='Number of training epochs')
    parser.add_argument('--checkpoint', type=str, help='Load checkpoint instead of training (path to .pkl file)')
    parser.add_argument('--seq-length', type=int, help='Sequence length for training and model')
    parser.add_argument('--text', type=str, help='Base text to repeat for dataset')
    parser.add_argument('--nbcopy', type=int, help='Number of times to repeat base text')
    parser.add_argument('--seed', type=str, help='Generation seed string')
    parser.add_argument('--temperature', type=float, help='Generation temperature (0 for argmax)')
    parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    args = parser.parse_args()

    # Use python-dotenv if available for richer parsing
    try:
        from dotenv import dotenv_values
        dotenv = dotenv_values(proj_root / '.env') or {}
    except Exception:
        dotenv = {}

    def get_conf(key, arg_val, env_key, default, kind=str):
        if arg_val is not None:
            return arg_val
        if env_key in os.environ:
            return coerce(os.environ[env_key], kind)
        if env_key in dotenv and dotenv[env_key] is not None:
            return coerce(dotenv[env_key], kind)
        return default

    num_epochs = get_conf('num_epochs', args.num_epochs, 'NUM_EPOCHS', 5, int)
    seq_length = get_conf('seq_length', args.seq_length, 'SEQ_LENGTH', 16, int)
    base_text = get_conf('text', args.text, 'BASE_TEXT', 'hello world ', str)
    nbcopy = get_conf('nbcopy', args.nbcopy, 'NB_COPY', 8, int)
    seed = get_conf('seed', args.seed, 'SEED', 'hello ', str)
    temperature = get_conf('temperature', args.temperature, 'TEMPERATURE', 0.0, float)
    batch_size = args.batch_size

    # Check if loading checkpoint instead of training
    checkpoint_path = args.checkpoint or dotenv.get('CHECKPOINT')
    
    if checkpoint_path:
        # Load model from checkpoint
        import pickle
        print(f'Loading checkpoint from {checkpoint_path}...')
        try:
            with open(checkpoint_path, 'rb') as f:
                ckpt = pickle.load(f)
            model = ckpt.get('model')
            if model is None:
                print('Error: checkpoint does not contain a `model` object.')
                return
            print('Checkpoint loaded successfully!')
            # Use tokenizer from checkpoint if available, else create one
            if 'tokenizer' in ckpt:
                tokenizer = ckpt['tokenizer']
            else:
                # Infer safe vocab size from model embedding and create fallback tokenizer
                vocab_size = int(model.word_embedding.shape[0])
                if vocab_size > 0 and vocab_size <= 95:
                    fallback_text = ''.join(chr(i) for i in range(32, 32 + vocab_size))
                    tokenizer = CharTokenizer(fallback_text)
                else:
                    tokenizer = CharTokenizer()
        except Exception as e:
            print(f'Failed to load checkpoint: {e}')
            return
    else:
        # Build dataset for training
        text = base_text * int(nbcopy)
        tokenizer = CharTokenizer(text)
        data = tokenizer.encode(text)

        from src.data_preprocessing import TextDataset, DataLoader
        dataset = TextDataset(data, int(seq_length))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Create model
        model = Transformer(vocab_size=tokenizer.vocab_size, embed_size=32, num_layers=1, heads=4, max_length=int(seq_length))

        # avoid thermal sleeps unless user configured otherwise
        os.environ.setdefault('THERMAL_TEMP_THRESHOLD', '1000')
        os.environ.setdefault('THERMAL_USAGE_THRESHOLD', '1000')

        print(f'Training tiny model for {num_epochs} epochs...')
        train_model(model, loader, num_epochs=int(num_epochs), save_path='models/', learning_rate=0.001)

    gen_len = max(1, 9)
    max_len = len(seed) + gen_len
    print('\n=== Generated output ===')
    out = generate_text(model, tokenizer, seed, max_length=max_len, temperature=float(temperature), seq_length=int(seq_length))
    print(out)


if __name__ == '__main__':
    main()
