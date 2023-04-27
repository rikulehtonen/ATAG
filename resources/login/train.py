import sys
import os
sys.path.insert(0, '../../')

from atag import Atag
from browserenv import BrowserEnv


def main():
    episodes = 10000
    browserEnv = BrowserEnv(collectData=True, resourcePath='')
    atag_browser = Atag(env=browserEnv, lr=0.001, gamma=0.99, model=None)

    atag_browser.train(episodes)
    browserEnv.terminate()

if __name__ == '__main__':
    main()