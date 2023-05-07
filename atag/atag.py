import sys, os
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import numpy as np

from .agent import PG

def createFolders(path):
    if not os.path.exists(path):
        os.makedirs(path)

def to_numpy(tensor):
    return tensor.squeeze(0).cpu().detach().numpy()

class Atag:
    def __init__(self, env, lr, gamma, model=None):
        self.env = env
        self.agent = PG(env.state_dim, env.action_dim, lr, gamma)
        self.agent.load(model)
        createFolders(self.env.resourcePath + 'results/model/')

    def run_episode(self, evaluation=False):
        reward_sum, timesteps, done = 0, 0, False
        obs, _, _ = self.env.reset()

        while not done:
            action, act_logprob = self.agent.get_action(obs, evaluation)
            obs, reward, done = self.env.step(to_numpy(action))
            if not evaluation:
                self.agent.record(act_logprob, reward)
            else:
                print(self.env.get_selected_action(to_numpy(action)))
            reward_sum += reward
            timesteps += 1

        # Update the policy after one episode
        if not evaluation:
            info = self.agent.update()
            # Return stats of training
            info.update({'timesteps': timesteps,
                        'ep_reward': reward_sum,})
            return info
        
        return {}

    def train(self, episodes):
        for ep in range(episodes):
            # collect data and update the policy
            train_info = self.run_episode()
            
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