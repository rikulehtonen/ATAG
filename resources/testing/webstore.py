from Browser import Browser
from Browser.utils.data_types import SupportedBrowsers
import time
import os
from Browser import AssertionOperator

def parse_keyword(target):
    keyword, args = target['keyword'], target['args']
    if "AssertionOperator" in args[1]:
        args[1] = getattr(AssertionOperator, args[1].split('.')[1])
    return keyword, args



b = Browser(timeout="5000 ms", retry_assertions_for="500 ms", strict=False)
b.new_browser(headless=False, browser=SupportedBrowsers.chromium)
b.new_context(
    acceptDownloads=True,
    viewport={"width": 700, "height": 500}
)
b.new_page('http://localhost:3000/category/xbox')
time.sleep(2)
b.click("xpath=//BUTTON[@class='ProductCard_button__vt_QY'][contains(text(),'Add to Cart')]")
time.sleep(5)

#print(b.get_element_states("//*[starts-with(., 'Grand Total: $') and number(substring-after(., 'Grand Total: $ ')) > 50]"))
target = {"keyword": "get_element_states", "args": ["//*[starts-with(., 'Grand Total: $') and number(substring-after(., 'Grand Total: $ ')) > 50]", "AssertionOperator.contains", "visible"], "positive_reward": 20.0, "negative_reward": 0, "is_done": True}
keyword, args = parse_keyword(target)
getattr(b, keyword)(*args, **{})

b.close_browser()

#getattr(b, 'type_text')('xpath=//input[@name="psw"]', 'testi')