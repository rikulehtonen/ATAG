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
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        self.log_std = torch.as_tensor(np.ones(action_dim, dtype=float) * 2.0)
        self.log_std = torch.nn.Parameter(torch.as_tensor(self.log_std))

    def forward(self, state):
        if not isinstance(state, torch.Tensor): state = torch.tensor(state, dtype=torch.float)
        return self.nn(state)