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


class Atag:
    def __init__(self, env, **parameters):
        self.env = env
        self.agent = PPO(env, env.state_dim, env.action_dim, **parameters)
        #self.agent.load(model)
        createFolders(self.env.resourcePath + 'results/model/')


    def train(self, episodes):
        for ep in range(episodes):
            # collect data and update the policy
            train_info = self.agent.run_episode()
            
            # Update results
            if (ep+1) % 100 == 0:
                self.agent.save(self.env.resourcePath + 'results/model/' + f'episode_{ep+1}_params.pt')

            train_info.update({'episodes': ep})
            print({"ep": ep, **train_info})


    def test(self, trials, path):
        for ep in range(trials):
            # collect data and update the policy
            self.agent.load(path)
            train_info = self.run_episode(evaluation=True)

            train_info.update({'episodes': ep})
            print({"ep": ep, **train_info})    