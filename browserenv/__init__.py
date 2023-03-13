from Browser import Browser
from Browser.utils.data_types import SupportedBrowsers
import time
import numpy as np
from Browser import AssertionOperator



class BrowserEnv:
    def __init__(self):
        self.b = Browser(timeout="20 s", retry_assertions_for="500 ms")
        self.b.new_browser(headless=False, browser=SupportedBrowsers.chromium)
        self.b.new_context(
            acceptDownloads=True,
            viewport={"width": 700, "height": 500}
        )

        self.steps = 0

        self.targets = [['get_element_states', ['xpath=//div[@id="logininfo"]', AssertionOperator.contains, 'visible'], 1, 0]]

        self.elements = [['get_element_states', ['xpath=//form[@id="myForm"]', AssertionOperator.contains, 'visible']],
                         ['get_element_states', ['xpath=//div[@id="loginFailed"]', AssertionOperator.contains, 'visible']],
                         ['get_element_states', ['xpath=//div[@id="logininfo"]', AssertionOperator.contains, 'visible']]]

        self.actions = [['click', ['xpath=//button[@id="loginBox"]']],
                        ['click', ['xpath=//button[@type="submit"]']],
                        ['type_text', ['xpath=//input[@name="uname"]','testaaja']],
                        ['type_text', ['xpath=//input[@name="psw"]','testi']]]

    def state_dim(self):
        return self.elements.len()
    
    def action_dim(self):
        return self.actions.len()

    def reset(self):
        self.b.close_page()
        self.b.new_page("file:///Users/riku/Documents/Aalto/ATAG/resources/login/login.html")
        self.steps = 0

    def terminate(self):
        self.b.close_browser()
    
    def action(self, act, args, kwargs):
        try:
            getattr(self.b, act)(*args, **kwargs)
            return None
        except:
            return -1

    def observe(self):
        done = False
        reward = None
        obs = np.zeros(self.elements.len())

        for i in range(self.elements.len()):
            try:
                getattr(self.b, self.elements[0])(*self.elements[1], **{})
                obs[i] = 1
            except:
                obs[i] = 0

        for i in range(self.targets.len()):
            try:
                getattr(self.b, self.targets[0])(*self.targets[1], **{})
                reward = self.targets[2]
            except:
                reward = self.targets[3]

        if self.steps > 10:
            done = True

        return obs, reward, done

    def step(self, act):
        #TODO: FIX LIST
        print(act)
        time.sleep(5)
        selected_act = self.actions[0]
        act_r = self.action(self, selected_act[0], selected_act[1], {})
        obs, reward, done = self.observe()
        self.steps += 1
        return obs, np.min(reward, act_r), done