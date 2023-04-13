from Browser import Browser
from Browser.utils.data_types import SupportedBrowsers
import time
import numpy as np
from Browser import AssertionOperator
import os
from .observer import Observer
from .datahandler import DataLoad, DataSave

class BrowserEnv:
    def __init__(self, collectData=False, folder='config/'):
        self.b = Browser(timeout="1.2 s", retry_assertions_for="500 ms")
        self.b.new_browser(headless=False, browser=SupportedBrowsers.chromium)
        self.b.new_context(
            acceptDownloads=True,
            viewport={"width": 700, "height": 500}
        )

        self.collectData = collectData

        self.load = DataLoad(folder)
        self.save = DataSave(folder)
        self.action_dim = self.load.lenActions()
        self.state_dim = self.load.lenElements()

        self.observer = Observer(self.b, self.collectData, self.load, self.save)
        self.init_steps()


    def init_steps(self):
        # Todo: Create Initializer
        page = 'file://' + os.getcwd() + '/resources/login/login.html'
        self.b.new_page(page)


    def reset(self):
        self.b.close_page()
        self.init_steps()
        self.observer.reset()
        return self.observer.observe()


    def terminate(self):
        self.b.close_browser()


    def take_action(self, act, args, kwargs):
        try:
            getattr(self.b, act)(*args, **kwargs)
            return -0.1
        except:
            return -1.0


    def step(self, act):
        selected_act = self.load.get_action(act.argmax())
        act_r = self.take_action(selected_act['keyword'], selected_act['args'], {})
        obs, reward, done = self.observer.observe()

        if reward != None:
            return obs, np.max([reward, act_r]), done
        else:
            return obs, act_r, done