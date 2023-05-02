import sys, os
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import numpy as np

def to_numpy(tensor):
    return tensor.squeeze(0).cpu().detach().numpy()

def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# Use CUDA for storing tensors / calculations if it's available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialisation function for neural network layers
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def createFolders(path):
    if not os.path.exists(path):
        os.makedirs(path)

# This class defines the neural network policy
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()

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
    

class PG(object):
    def __init__(self, state_dim, action_dim, lr, gamma):

        self.policy = Policy(state_dim, action_dim).to(device)
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
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
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
