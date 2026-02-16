"""
Discret Test Module

Eval line by line the discretization of the codebase.
"""
import torch
import torch.nn as nn
from torch.optim import Adam


def main():
    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
main()