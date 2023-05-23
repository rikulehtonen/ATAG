import sys
import os
sys.path.insert(0, '../../')

from atag import Atag
from browserenv import BrowserEnv
from config.atag_config import Atag_config

#Parameters for Atag & algorithm
parameters = {
    'lr': 1e-3,
    'max_timesteps': 5,
    'batch_timesteps': 1,
    'episode_max_timesteps': 2,
    'iteration_epochs': 1,
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
    atag_browser.test()
    browserEnv.terminate()

if __name__ == '__main__':
    main()