from Browser import Browser
from Browser.utils.data_types import SupportedBrowsers
import time
import os

b = Browser(timeout="1000 ms", retry_assertions_for="500 ms", strict=False)
b.new_browser(headless=False, browser=SupportedBrowsers.chromium)
b.new_context(
    acceptDownloads=True,
    viewport={"width": 700, "height": 500}
)
b.new_page('http://localhost:3000/category/xbox')
time.sleep(2)
b.click("xpath=//BUTTON[@class='ProductCard_button__vt_QY'][contains(text(),'Add to Cart')]")
time.sleep(5)
b.close_browser()

#getattr(b, 'type_text')('xpath=//input[@name="psw"]', 'testi')