from Browser import Browser
from Browser.utils.data_types import SupportedBrowsers
import time
import numpy as np
import os

class Atag_config(object):
    def __init__(self):
        self.test_env = None

        self.env_parameters = {
            'elements_file': 'config_elements.json',
            'actions_file': 'config_actions.json',
            'config_path': 'config/',
            'results_location': 'results/model/',
            'passed_action_cost': -3.0,
            'failed_action_cost': -10.0
        }

        self.data_collection = {
            'collect_data': False,
            'collect_path': False,
            'collect_path_file': 'results/path.json',
            'elements_file': 'config_elements.json',
            'actions_file': 'config_actions.json',
            'temp_config_path': 'config/temp/',
            'click_actions': ['A', 'BUTTON'],
            'ignore_elements': [],
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
        page = 'file://' + os.getcwd() + '/button.html'
        self.test_env.set_browser_timeout("10 s")
        self.test_env.new_page(page)
        self.test_env.set_browser_timeout("700 ms")

    def teardown_test(self):
        self.test_env.close_page()

    def env_ready(self):
        time.sleep(0.02)

    def state_rewards(self):

        xpath = "//p[contains(text(),'Error')]"
        if "visible" in self.test_env.get_element_states(xpath):
            return 800, False

        xpath = "//p[@id='modalContent']"
        try:
            points = self.test_env.get_text(xpath)
            return int(points), False
        except:
            return 0, False

