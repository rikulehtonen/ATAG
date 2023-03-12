from Browser import Browser
from Browser.utils.data_types import SupportedBrowsers
import time



class BrowserEnv:
    def __init__(self):
        self.b = Browser(timeout="20 s", retry_assertions_for="500 ms")
        self.b.new_browser(headless=False, browser=SupportedBrowsers.chromium)
        self.b.new_context(
            acceptDownloads=True,
            viewport={"width": 700, "height": 500}
        )

    def reset(self):
        self.b.close_page()
        self.b.new_page("file:///Users/riku/Documents/Aalto/ATAG/resources/login/login.html")

    def terminate(self):
        self.b.close_browser()
    
    def step(self):

        return None