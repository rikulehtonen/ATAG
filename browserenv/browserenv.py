from Browser import Browser
from Browser.utils.data_types import SupportedBrowsers
import time
import numpy as np
from Browser import AssertionOperator
import os
from .observer import Observer
from .datahandler import DataLoad, DataSave

class BrowserEnv:
    def __init__(self, collectData=False, resourcePath=''):
        self.b = Browser(timeout="200 ms", retry_assertions_for="60 ms")
        self.b.new_browser(headless=False, browser=SupportedBrowsers.chromium)
        self.b.new_context(
            acceptDownloads=True,
            viewport={"width": 700, "height": 500}
        )

        self.resourcePath = resourcePath
        self.collectData = collectData

        self.load = DataLoad(self.resourcePath + 'config/')
        self.save = DataSave(self.resourcePath + 'config/')
        self.action_dim = self.load.lenActions()
        self.state_dim = self.load.lenElements()

        self.observer = Observer(self.b, self.collectData, self.load, self.save)
        self.init_steps()

    def init_steps(self):
        # TODO: Create Initializer
        page = 'file://' + os.getcwd() + '/login.html'
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
            return -3.0
        except:
            return -10.0

    def get_selected_action(self, act):
        return self.load.get_action(act.argmax())

    def step(self, act):
        selected_act = self.get_selected_action(act)
        act_reward = self.take_action(selected_act['keyword'], selected_act['args'], {})
        obs, obs_reward, done = self.observer.observe()
        reward = act_reward + obs_reward

        return obs, reward, done
