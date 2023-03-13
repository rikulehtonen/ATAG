from Browser import Browser
from Browser.utils.data_types import SupportedBrowsers
import time
from Browser import AssertionOperator

b = Browser(timeout="20 s", retry_assertions_for="500 ms")
b.new_browser(headless=False, browser=SupportedBrowsers.chromium)
b.new_context(
    acceptDownloads=True,
    viewport={"width": 700, "height": 500}
)
b.new_page("file:///Users/riku/Documents/Aalto/ATAG/resources/login/login.html")
print(b.get_element_states('xpath=//form[@id="myForm"]'))
print(b.get_element_states('xpath=//form[@id="myForm"]', AssertionOperator.contains, 'visible'))
b.close_browser()

#getattr(b, 'type_text')('xpath=//input[@name="psw"]', 'testi')