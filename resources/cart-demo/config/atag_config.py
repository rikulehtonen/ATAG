from Browser import Browser
from Browser.utils.data_types import SupportedBrowsers
import time
import numpy as np

class Atag_config(object):
    def __init__(self):
        self.test_env = None

        self.env_parameters = {
            'elements_file': 'config_elements.json',
            'actions_file': 'config_actions.json',
            'config_path': 'config/',
            'results_location': 'results/model/',
            'passed_action_cost': -10.0,
            'failed_action_cost': -20.0,
        }

        self.data_collection = {
            'collect_data': True,
            'elements_file': 'config_elements.json',
            'actions_file': 'config_actions.json',
            'temp_config_path': 'config/temp/',
            'click_actions': ['A', 'BUTTON'],
            'ignore_elements': ['DIV'],
            'type_actions': ['INPUT'],
            'type_word_list': ['testaaja', 'testi', 'salasana']
        }

    def setup_env(self):
        self.test_env = Browser(timeout="10000 ms", retry_assertions_for="10 ms", strict=False)
        self.test_env.new_browser(headless=False, browser=SupportedBrowsers.chromium)
        self.test_env.new_context(acceptDownloads=True, viewport={"width": 700, "height": 500})
        return self.test_env
    
    def teardown_env(self):
        self.test_env.close_browser()

    def setup_test(self):
        page = 'http://localhost:3000/'
        self.test_env.set_browser_timeout("10 s")
        self.test_env.new_page(page)
        self.test_env.set_browser_timeout("100 ms")

    def teardown_test(self):
        self.test_env.close_page()

    def env_ready(self):
        time.sleep(0.7)

    def state_rewards(self):
        reward = 0.0
        done = False

        xpath = "//*[starts-with(., 'Grand Total: $') and number(substring-after(., 'Grand Total: $ ')) > 100]"
        if "visible" in self.test_env.get_element_states(xpath):
            reward += 90.0
            done = True

        xpath = "//*[starts-with(., 'Grand Total: $') and number(substring-after(., 'Grand Total: $ ')) > 50]"
        if "visible" in self.test_env.get_element_states(xpath):
            reward += 60.0
            done = True

        xpath = "//div[@class='CartPage_body__9xgUX']//*[contains(text(),'Fortnite')]"
        if "visible" in self.test_env.get_element_states(xpath):
            reward += 40.0
        
        xpath = "//A[contains(text(),'Cart') and not(contains(text(),'Cart (0)'))]"
        if "visible" in self.test_env.get_element_states(xpath):
            reward += 6.0

        return reward, done
