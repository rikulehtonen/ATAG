import sys
import os
sys.path.insert(0, '../../')

from atag import Atag
from browserenv import BrowserEnv


parameters = {
    'lr': 1e-4,
    'max_timesteps': 200000000,
    'batch_timesteps': 6,
    'episode_max_timesteps': 12,
    'iteration_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip': 0.2,
    'save_frequency': 50
}

def main():
    episodes = 10000
    browserEnv = BrowserEnv(collectData=True, resourcePath='')
    atag_browser = Atag(env=browserEnv, **parameters)

    atag_browser.train(episodes)
    browserEnv.terminate()

if __name__ == '__main__':
    main()