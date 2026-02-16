# LLM from Scratch - Pedagogical Project

This project is an educational implementation of building a Large Language Model (LLM) from scratch using Python and PyTorch. The goal is to understand the fundamentals of transformer-based models, including data preprocessing, model architecture, training, and evaluation.

## Project Structure

- `src/`: Main source code
  - `data_preprocessing.py`: Tokenization and data loading
  - `model.py`: Transformer model implementation
  - `training.py`: Training loop and optimization
  - `evaluation.py`: Model evaluation and metrics
  - `main.py`: Entry point to run the project
- `data/`: Directory for datasets
- `models/`: Saved model checkpoints
- `utils/`: Utility functions
- `tests/`: Unit tests
- `notebooks/`: Jupyter notebooks for experimentation

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt` or `pip install -e .`

## Usage

Run the main script: `python -m src.main`

Or explore the notebooks in the `notebooks/` directory.

## Learning Objectives

- Understand tokenization and text preprocessing
- Implement self-attention mechanism
- Build a transformer decoder
- Train a language model on text data
- Evaluate generation quality

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU recommended for training

## Disclaimer

This is a simplified implementation for educational purposes. Real-world LLMs require massive datasets and computational resources.