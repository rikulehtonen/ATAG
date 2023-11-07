import sys, os
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import numpy as np
from .nn import NeuralNet
import wandb
import time
import random

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
        self.trainingData = params.get('trainingData')

        self.actor = NeuralNet(state_dim, action_dim)
        self.critic = NeuralNet(state_dim, 1)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.params.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.params.lr)

        self.env = env
        self.action_probs = []
        self.rewards = []
        if params.get('log_to_wandb'):
            wandb.init(project="Thesis-results", entity="rikulehtonen", group=params.get('name'))
        self.start_time = time.time()


    def run_episode(self, evaluation=False):
        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_entropies = []

        for batch_iterations in range(self.params.batch_timesteps):
            
            ep_obs = []
            ep_next_obs = []
            ep_actions = []
            ep_act_probs = []
            ep_rewards = []
            ep_dones = []

            obs = self.env.reset()
            done = False

            for total_iterations in range(self.params.episode_max_timesteps):
                ep_obs.append(obs)
                batch_obs.append(obs)
                action, act_logprob, act_probs, entropy = self.get_action(obs, evaluation)
                obs, reward, done, _ = self.env.step(action, evaluation)

                ep_next_obs.append(obs)
                ep_actions.append(action)
                ep_act_probs.append(act_probs)
                ep_rewards.append(reward)
                ep_dones.append(done)

                batch_actions.append(action)
                batch_log_probs.append(act_logprob)
                batch_entropies.append(entropy)

                if done: break

            if self.trainingData:
                self.trainingData.save(ep_obs,ep_next_obs,ep_actions,ep_act_probs,ep_rewards,ep_dones)

            batch_rewards.append(ep_rewards)

            batch_obs_s = torch.tensor(batch_obs, dtype=torch.float)
            batch_actions_s = torch.tensor(batch_actions, dtype=torch.float)
            batch_log_probs_s = torch.tensor(batch_log_probs, dtype=torch.float)
            batch_entropies_s = torch.tensor(batch_entropies, dtype=torch.float)

            V, _ = self.get_value(batch_obs_s, batch_actions_s)

            A = self.generalized_advantage_estimate(batch_rewards, V)
            
            R = A + V.detach() 
            
            division = 3

            batch_obs_s = torch.split(batch_obs_s, division)
            batch_actions_s = torch.split(batch_actions_s, division)
            batch_log_probs_s = torch.split(batch_log_probs_s, division)
            batch_entropies_s = torch.split(batch_entropies_s, division)
            
            V = torch.split(V, division)
            A = torch.split(A, division)
            R = torch.split(R, division)

            inds = np.arange(round(len(A) / division) - 1)
            np.random.shuffle(inds)

            for _ in range(self.params.iteration_epochs):
                for mini_batch in inds:

                    # Add entropy bonus to actor loss

                    AM = (A[mini_batch] - A[mini_batch].mean()) / (A[mini_batch].std() + 1e-10)
                    V, curr_log_probs = self.get_value(batch_obs_s[mini_batch], batch_actions_s[mini_batch])

                    ratio = torch.exp(curr_log_probs - batch_log_probs_s[mini_batch])
                    
                    # Original actor loss
                    actor_loss = (-torch.min(ratio * AM, torch.clamp(ratio, 1 - self.params.clip, 1 + self.params.clip) * AM)).mean()

                    # Add the entropy bonus, scaled by the coefficient
                    entropy_bonus = (self.params.entropy_coeff * batch_entropies_s[mini_batch]).mean()

                    # Total actor loss including the entropy bonus
                    total_actor_loss = actor_loss - entropy_bonus  # Subtracting because we’re minimizing the loss

                    critic_loss = nn.MSELoss()(V, R[mini_batch])

                    self.actor_optimizer.zero_grad()
                    total_actor_loss.backward(retain_graph=True)  # Updated to total_actor_loss
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.actor_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.critic_optimizer.step()
            
            ep_reward = np.mean([np.sum(ep_rewards) for ep_rewards in batch_rewards])
            if self.params.get('log_to_wandb'):
                wandb.log({"ep_reward": ep_reward, "time_d": (time.time() - self.start_time), "is_done": (float(done))})

        return {'timesteps': self.params.batch_timesteps, 'ep_reward': ep_reward}


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

        advantages = torch.tensor(advantages, dtype=torch.float)
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def get_action(self, state, evaluation):

        probs = self.actor(state, 1.2)  # Assuming actor returns a probability distribution
        dist = torch.distributions.Categorical(probs)

        if evaluation:
            action = torch.argmax(probs).item()
            return action
                
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action)) + 1e-8

        return action, log_prob.detach(), probs.detach(), dist.entropy().detach()

    def get_value(self, batch_state, batch_actions):
        V = self.critic(batch_state).squeeze()

        action_probs = self.actor(batch_state)
        m = torch.distributions.Categorical(action_probs)
        log_probs = m.log_prob(batch_actions)

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

    def evaluate(self):
        rewards = []
        for batch_iterations in range(self.params.batch_timesteps):
            obs = self.env.reset()
            rewardSum = 0
            for total_iterations in range(self.params.episode_max_timesteps):
                action, act_logprob, act_probs, entropy = self.get_action(obs, False)
                obs, reward, done, _ = self.env.step(action)
                rewardSum += reward
                if done: break
            rewards.append(rewardSum)

        return rewards