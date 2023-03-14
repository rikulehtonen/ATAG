from atag import Atag
from browserenv import BrowserEnv

def load_conf():
    pass

def main():
    episodes = 100
    browserEnv = BrowserEnv()
    atag_browser = Atag(browserEnv,0.1,0.1)
    atag_browser.train(episodes)
    browserEnv.terminate()

if __name__ == '__main__':
    main()