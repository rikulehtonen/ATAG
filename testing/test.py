from Browser import Browser
from Browser.utils.data_types import SupportedBrowsers
import time

b = Browser(timeout="20 s", retry_assertions_for="500 ms")
b.new_browser(headless=False, browser=SupportedBrowsers.chromium)
b.new_context(
    acceptDownloads=True,
    viewport={"width": 700, "height": 500}
)
b.new_page("file:///Users/riku/Documents/Aalto/ATAG/resources/login/login.html")
b.click('xpath=//button[@id="loginBox"]')
b.type_text('xpath=//input[@name="uname"]',"testaaja")
b.type_text('xpath=//input[@name="psw"]',"testi")
print(b.get_text('xpath=//input[@name="uname"]'))
b.click('xpath=//button[@type="submit"]')
assert b.get_text('xpath=//*[@id="logininfo"]') == 'Logged In'
b.close_browser()

#getattr(b, 'type_text')('xpath=//input[@name="psw"]', 'testi')