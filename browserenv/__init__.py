from Browser import Browser
from Browser.utils.data_types import SupportedBrowsers
import time
import numpy as np
from Browser import AssertionOperator



class BrowserEnv:
    def __init__(self):
        self.b = Browser(timeout="1.2 s", retry_assertions_for="500 ms")
        self.b.new_browser(headless=False, browser=SupportedBrowsers.chromium)
        self.b.new_context(
            acceptDownloads=True,
            viewport={"width": 700, "height": 500}
        )
        self.b.new_page("file:///Users/riku/Documents/Aalto/ATAG/resources/login/login.html")

        self.steps = 0

        self.targets = [['get_element_states', ['xpath=//div[@id="logininfo"]', AssertionOperator.contains, 'visible'], 1.0, 0.0, True]]

        self.elements = [['get_element_states', ['xpath=//form[@id="myForm"]', AssertionOperator.contains, 'visible']],
                         ['get_element_states', ['xpath=//div[@id="loginFailed"]', AssertionOperator.contains, 'visible']],
                         ['get_element_states', ['xpath=//div[@id="logininfo"]', AssertionOperator.contains, 'visible']]]

        self.actions = [['click', ['xpath=//button[@id="loginBox"]']],
                        ['click', ['xpath=//button[@type="submit"]']],
                        ['type_text', ['xpath=//input[@name="uname"]','testaaja']],
                        ['type_text', ['xpath=//input[@name="psw"]','testi']]]

    def state_dim(self):
        return len(self.elements)
    
    def action_dim(self):
        return len(self.actions)

    def reset(self):
        self.b.close_page()
        self.b.new_page("file:///Users/riku/Documents/Aalto/ATAG/resources/login/login.html")
        self.steps = 0
        return self.observe()

    def terminate(self):
        self.b.close_browser()
    
    def action(self, act, args, kwargs):
        try:
            getattr(self.b, act)(*args, **kwargs)
            return 0
        except:
            return -1

    def observe(self):
        done = False
        reward = None
        obs = np.zeros(self.state_dim())

        for i in range(self.state_dim()):
            try:
                getattr(self.b, self.elements[i][0])(*self.elements[i][1], **{})
                obs[i] = 1.0
            except:
                obs[i] = 0.0

        for id in range(len(self.targets)):
            try:
                getattr(self.b, self.targets[id][0])(*self.targets[id][1], **{})
                reward = self.targets[id][2]
                done = self.targets[id][4]
                if done: break
            except:
                reward = self.targets[id][3]

        if self.steps > 5:
            done = True

        return obs, reward, done


    def step(self, act):
        selected_act = self.actions[act.argmax()]
        act_r = self.action(selected_act[0], selected_act[1], {})
        obs, reward, done = self.observe()

        self.steps += 1
        return obs, np.min([reward, act_r]), done