import sys, os
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import numpy as np
from .nn import NeuralNet
#import wandb
import time

# Use CUDA for storing tensors / calculations if it's available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_numpy(tensor):
    return tensor.squeeze(0).cpu().detach().numpy()

def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class PPO(object):
    def __init__(self, env, state_dim, action_dim, params):
        self.params = params

        self.actor = NeuralNet(state_dim, action_dim)
        self.critic = NeuralNet(state_dim, 1)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.params.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.params.lr)

        self.env = env
        self.action_probs = []
        self.rewards = []

        #wandb.init(project="ATAG", entity="rikulehtonen")
        self.start_time = time.time()


    def run_episode(self, evaluation=False):
        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []

        for batch_iterations in range(self.params.batch_timesteps):
            ep_rewards = []
            obs, _, _ = self.env.reset()
            done = False

            for total_iterations in range(self.params.episode_max_timesteps):
                batch_obs.append(obs)
                action, act_logprob = self.get_action(obs, evaluation)
                obs, reward, done = self.env.step(action, evaluation)

                ep_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(act_logprob)

                if done: break

            batch_rewards.append(ep_rewards)

            batch_obs_s = torch.tensor(batch_obs, dtype=torch.float)
            batch_actions_s = torch.tensor(batch_actions, dtype=torch.float)
            batch_log_probs_s = torch.tensor(batch_log_probs, dtype=torch.float)

            V, _ = self.get_value(batch_obs_s, batch_actions_s)

            A = self.generalized_advantage_estimate(batch_rewards, V)
            
            R = A + V.detach() 
            
            division = 3

            batch_obs_s = torch.split(batch_obs_s, division)
            batch_actions_s = torch.split(batch_actions_s, division)
            batch_log_probs_s = torch.split(batch_log_probs_s, division)
            
            V = torch.split(V, division)
            A = torch.split(A, division)
            R = torch.split(R, division)

            inds = np.arange(round(len(A) / division) - 1)
            np.random.shuffle(inds)

            for _ in range(self.params.iteration_epochs):
                for mini_batch in inds:
                    
                    AM = (A[mini_batch] - A[mini_batch].mean()) / (A[mini_batch].std() + 1e-10)
                    V, curr_log_probs = self.get_value(batch_obs_s[mini_batch], batch_actions_s[mini_batch])

                    ratio = torch.exp(curr_log_probs - batch_log_probs_s[mini_batch])
                    actor_loss = (-torch.min(ratio * AM, torch.clamp(ratio, 1 - self.params.clip, 1 + self.params.clip) * AM)).mean()
                    critic_loss = nn.MSELoss()(V, R[mini_batch])

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.actor_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.critic_optimizer.step()
            
            ep_reward = np.mean([np.sum(ep_rewards) for ep_rewards in batch_rewards])
            #wandb.log({"ep_reward": ep_reward, "time_d": (time.time() - self.start_time), "is_done": (float(done))})

        return {'timesteps': 0, 'ep_reward': ep_reward}


    def generalized_advantage_estimate(self, batch_rewards, V):
        i = len(V) - 1
        advantages = []
        
        for ep_rewards in reversed(batch_rewards):
            advantage = 0
            next_value = 0

            for reward in reversed(ep_rewards):
                td = reward + next_value * self.params.gamma - V[i]
                advantage = td + advantage * self.params.gamma * self.params.gae_lambda
                next_value = V[i]
                advantages.insert(0, advantage)
                i -= 1

        return torch.tensor(advantages, dtype=torch.float)

    def get_action(self, state, evaluation):
        mean = self.actor(state)

        if evaluation:
            return  mean.detach().numpy(), 1
        
        dist = torch.distributions.Normal(mean, self.actor.log_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)

        return action.detach().numpy(), log_prob.detach()

    def get_value(self, batch_state, batch_actions):
        V = self.critic(batch_state).squeeze()

        mean = self.actor(batch_state)
        dist = torch.distributions.Normal(mean, self.actor.log_std)
        log_probs = dist.log_prob(batch_actions).sum(axis=-1)

        return V, log_probs


    def save(self, filepath, total_iterations):
        if filepath != None:
            torch.save(self.actor.state_dict(), f'{filepath}{total_iterations}_actor.pt')
            torch.save(self.critic.state_dict(), f'{filepath}{total_iterations}_critic.pt')

    def load(self):
        actor_file = self.params.actor_file
        critic_file = self.params.critic_file
        if actor_file != None:
            self.actor.load_state_dict(torch.load(actor_file))
        if critic_file != None:
            self.critic.load_state_dict(torch.load(critic_file))

