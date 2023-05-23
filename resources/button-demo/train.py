import sys
import os
sys.path.insert(0, '../../')

from atag import Atag
from browserenv import BrowserEnv
from config.atag_config import Atag_config

#Parameters for Atag & algorithm
parameters = {
    'lr': 1e-3,
    'max_timesteps': 200000000,
    'batch_timesteps': 10,
    'episode_max_timesteps': 3,
    'iteration_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip': 0.2,
    'save_frequency': 50,
    'actor_file': 'actor.pt',
    'critic_file': 'critic.pt'
}

def main():
    # Config for browserEnvironment
    config = Atag_config()
    # Test environment for Browser library
    browserEnv = BrowserEnv(config)
    # ML algorithm for test generation
    atag_browser = Atag(env=browserEnv, **parameters)

    # Train 
    atag_browser.train()
    browserEnv.terminate()

if __name__ == '__main__':
    main()