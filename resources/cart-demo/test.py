import sys
import os
sys.path.insert(0, '../../')

from atag import Atag
from browserenv import BrowserEnv

def main():
    browserEnv = BrowserEnv()
    atag_browser = Atag(browserEnv, 0.01, 0.99)

    atag_browser.test(1, 'model.pt')

    browserEnv.terminate()

if __name__ == '__main__':
    main()