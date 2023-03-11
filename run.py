from atag import Atag
from atag import BrowserHelper

import sys, os
import time
from pathlib import Path

import torch
import gym
import hydra
import wandb
import warnings


#def main():
#    Helper = BrowserHelper()
#    atag_browser = Atag(Helper)
#    atag_browser.run()

def to_numpy(tensor):
    return tensor.squeeze(0).cpu().numpy()


def train(agent, env):
    # Policy training function

    # Reset the environment and observe the initial state
    reward_sum, timesteps, done = 0, 0, False
    obs = env.reset()

    while not done:
        action, act_logprob = agent.get_action(obs)
        obs, reward, done, _ = env.step(to_numpy(action))
        agent.record(act_logprob, reward)
        reward_sum += reward
        timesteps += 1

    # Update the policy after one episode
    info = agent.update()

    # Return stats of training
    info.update({'timesteps': timesteps, 'ep_reward': reward_sum,})
    return info


# Function to test a trained policy
@torch.no_grad()
def test(agent, env, num_episodes=10):

    total_test_reward = 0
    for ep in range(num_episodes):
        obs, done= env.reset(), False
        test_reward = 0

        while not done:
            action, _ = agent.get_action(obs, evaluation=True)
            obs, reward, done, info = env.step(to_numpy(action))
            test_reward += reward

        total_test_reward += test_reward

        print("Test ep_reward:", test_reward)
    print("Average test reward:", total_test_reward/num_episodes)


# The main function
@hydra.main(config_path='cfg', config_name='ex5_cfg')
def main(cfg):

    # Set seed for random number generators
    h.set_seed(cfg.seed)

    # Define a run id based on current time
    cfg.run_id = int(time.time())

    # Create folders if needed
    work_dir = Path().cwd()/'results'
    if cfg.save_model: h.make_dir(work_dir/"model")
    if cfg.save_logging: 
        h.make_dir(work_dir/"logging")
        L = logger.Logger() # create a simple logger to record stats

    # Model filename
    if cfg.model_path == 'default':
        cfg.model_path = work_dir/'model'/f'{cfg.env_name}_params.pt'

    # Create the gym env
    env = gym.make(cfg.env_name, render_mode='rgb_array' if cfg.save_video else None)

    # Set env random seed
    env.seed(cfg.seed)

    # Get state and action dimensionality
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialise the policy gradient agent
    agent = PG(state_dim, action_dim, cfg.lr, cfg.gamma)

    if not cfg.testing: # training
        for ep in range(cfg.train_episodes):
            # collect data and update the policy
            train_info = train(agent, env)
            train_info.update({'episodes': ep})
            
            if cfg.use_wandb:
                wandb.log(train_info)
            if cfg.save_logging:
                L.log(**train_info)
            if (not cfg.silent) and (ep % 100 == 0):
                print({"ep": ep, **train_info})
        
        if cfg.save_model:
            agent.save(cfg.model_path)

    else: # testing
        print("Loading model from", cfg.model_path, "...")
        # load model
        agent.load(cfg.model_path)
        print('Testing ...')
        test(agent, env, num_episodes=10)




if __name__ == '__main__':
    main()