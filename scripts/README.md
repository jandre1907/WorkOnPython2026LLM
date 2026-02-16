Generate from checkpoint
========================

Quick instructions to run the small generator script included in this project.

Prerequisites
- A Python virtualenv with project dependencies (use the project's `.venv` if available).

Usage
-----
Run the script with a checkpoint produced by training. Example:

```bash
.venv/bin/python scripts/generate_from_checkpoint.py \
  --ckpt models/model_epoch_3.pkl \
  --prompt "Hello" \
  --length 120 \
  --temperature 1.0
```

Notes
- Checkpoints are unpickled directly; only load files you trust.
- If the checkpoint does not contain a tokenizer, the script will attempt a safe fallback tokenizer
  whose vocab size matches the model embedding.
