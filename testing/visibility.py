from Browser import Browser
from Browser.utils.data_types import SupportedBrowsers
import time
from Browser import AssertionOperator
import os

def parse_keyword(target):
    keyword, args = target['keyword'], target['args']
    if "AssertionOperator" in args[1]:
        args[1] = getattr(AssertionOperator, args[1].split('.')[1])
    return keyword, args


b = Browser(timeout="20 s", retry_assertions_for="500 ms")
b.new_browser(headless=False, browser=SupportedBrowsers.chromium)
b.new_context(
    acceptDownloads=True,
    viewport={"width": 700, "height": 500}
)
b.new_page('file://' + os.getcwd() + '/resources/login/login.html')
#print(b.get_element_states('xpath=//form[@id="myForm"]'))
#b.click('xpath=//button[@id="loginBox"]')
#print(b.get_element_states('xpath=//form[@id="myForm"]', AssertionOperator.contains, 'visible'))
#getattr(b, 'get_element_states')(*["xpath=//form[@id='myForm']", "AssertionOperator.contains", "visible"], **{})

target = {"keyword": "get_element_states", "args": ["xpath=//form[@id='myForm']", "AssertionOperator.contains", "visible"], "positive_reward": 3.0, "negative_reward": 0, "is_done": False}
keyword, args = parse_keyword(target)
getattr(b, keyword)(*args, **{})
time.sleep(3)
b.close_browser()



#getattr(b, 'type_text')('xpath=//input[@name="psw"]', 'testi')