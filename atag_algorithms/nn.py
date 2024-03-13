import sys, os
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import numpy as np

# Use CUDA for storing tensors / calculations if it's available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialisation function for neural network layers
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# This class defines the neural network policy
class NeuralNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(NeuralNet, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_dim),
        )

    def forward(self, state, temperature=1.0):
        if not isinstance(state, torch.Tensor): state = torch.tensor(state, dtype=torch.float)
        logits = self.nn(state)
        # Apply temperature scaling to the logits before softmax
        scaled_logits = logits / temperature
        return F.softmax(scaled_logits, dim=-1)
