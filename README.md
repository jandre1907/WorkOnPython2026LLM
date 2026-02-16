# LLM from Scratch - Pedagogical Project

This project is an educational implementation of building a Large Language Model (LLM) from scratch using **Python and NumPy only**. The goal is to understand the fundamentals of transformer-based models, including data preprocessing, model architecture, training, and evaluation—without relying on PyTorch or TensorFlow.

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
- NumPy >= 1.21.0
- Optional: psutil for CPU thermal monitoring

## Thermal Protection (CPU Monitoring)

This project includes a thermal guard (`utils/thermal_guard.py`) to monitor and throttle training when CPU temperature or usage exceeds safe thresholds. This prevents hardware damage during long training runs.

### Setup on Windows 10 + WSL2

For systems running Windows 10 with WSL2:

1. **Install LibreHardwareMonitor** on Windows 10 (host OS)
   - Download from: https://github.com/LibreHardwareMonitor/LibreHardwareMonitor
   - Run the application to monitor CPU/GPU temperatures

2. **Enable API endpoint**
   - LibreHardwareMonitor publishes CPU/temperature data on `0.0.0.0:8085`
   - From WSL2, access it via: `http://172.29.128.1:8085/data.json`
   - This endpoint is used by the thermal guard to read live temperature data

3. **Configure in your environment**
   - Set `THERMAL_API_URL=http://172.29.128.1:8085/data.json` in `.env` or as an environment variable
   - Adjust `THERMAL_TEMP_THRESHOLD` and `THERMAL_USAGE_THRESHOLD` as needed (defaults: 92°C, 85%)

### Fallback Monitoring

If LibreHardwareMonitor is unavailable:
- On Linux/Mac: uses `psutil.sensors_temperatures()`
- On all systems: falls back to `/sys/class/thermal/thermal_zone*/temp`

## Disclaimer

This is a simplified implementation for educational purposes. Real-world LLMs require massive datasets and computational resources.