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
            layer_init(nn.Linear(state_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, action_dim))
        )

        self.actor_logstd = torch.nn.Parameter(torch.tensor([0.1], device=device))
        #self.actor_logstd = torch.tensor([0.0], device=device)

    def forward(self, state):
        action_mean = self.nn(state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs