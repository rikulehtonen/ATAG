import sys, os
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import numpy as np

from .ppo import PPO

def createFolders(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Parameters(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Atag:
    def __init__(self, env, **parameters):
        self.params = Parameters(parameters)
        self.env = env
        self.agent = PPO(env, env.state_dim, env.action_dim, self.params)
        self.agent.load()
        createFolders(self.env.config.env_parameters.get('results_location'))


    def train(self):
        train_info = None
        for ep in range(self.params.max_timesteps):
            # collect data and update the policy
            train_info = self.agent.run_episode()
            
            # Update results
            if (ep+1) % 5 == 0:
                self.agent.save(self.env.config.env_parameters.get('results_location'), ep)

            train_info.update({'episodes': ep})
            print({"ep": ep, **train_info})

        return train_info.get('ep_reward')


    def evaluate(self):
        # collect data and update the policy
        return self.agent.evaluate()