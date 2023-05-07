import sys, os
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import numpy as np

from .nn import NeuralNet

# Use CUDA for storing tensors / calculations if it's available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def createFolders(path):
    if not os.path.exists(path):
        os.makedirs(path)


class PG(object):
    def __init__(self, state_dim, action_dim, lr, gamma):

        self.policy = NeuralNet(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.action_probs = []
        self.rewards = []

    def update(self,):
        # Prepare dataset used to update policy
        action_probs = torch.stack(self.action_probs, dim=0).to(device).squeeze(-1) # shape: [batch_size,]
        rewards = torch.stack(self.rewards, dim=0).to(device).squeeze(-1) # shape [batch_size,]
        self.action_probs, self.rewards = [], [] # clean buffers
        disc_rewards = discount_rewards(rewards, self.gamma)

        # Normalize rewards
        #disc_rewards=(disc_rewards - torch.mean(disc_rewards)) / torch.std(disc_rewards)
        baseline = 0
        loss = torch.mean(-(disc_rewards - baseline) * torch.t(action_probs)[0])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'logstd': self.policy.actor_logstd.cpu().detach().numpy()}

    def get_action(self, observation, evaluation=False):
        # if observation.ndim == 1: observation = observation[None]
        # print(observation)
        x = torch.from_numpy(observation).float().to(device)
        distrib=self.policy.forward(x)
        action = distrib.mean if evaluation else distrib.sample((1,))[0]
        act_logprob = distrib.log_prob(action)
        
        #if observation.ndim == 1: action = action[0]
        return action, act_logprob

    def record(self, action_prob, reward):
        """ Store agent's and env's outcomes to update the agent."""
        self.action_probs.append(action_prob)
        self.rewards.append(torch.tensor([reward]))

    def save(self, filepath):
        if filepath != None:
            torch.save(self.policy.state_dict(), filepath)

    def load(self, filepath):
        if filepath != None:
            self.policy.load_state_dict(torch.load(filepath))

