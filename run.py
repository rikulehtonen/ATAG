from atag import Atag
from browserenv import BrowserEnv


def main():
    episodes = 200
    browserEnv = BrowserEnv(collectData=True)
    atag_browser = Atag(browserEnv, 0.01, 0.99)
    atag_browser.train(episodes)
    browserEnv.terminate()

if __name__ == '__main__':
    main()