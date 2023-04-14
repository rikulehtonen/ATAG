from Browser import Browser
from Browser.utils.data_types import SupportedBrowsers
import time
import os
from Browser import AssertionOperator
import multiprocessing
import time

b = Browser(timeout="0 s", retry_assertions_for="0 ms")
b.new_browser(headless=False, browser=SupportedBrowsers.chromium)
b.new_context(
    acceptDownloads=True,
    viewport={"width": 700, "height": 500}
)
b.new_page('file://' + os.getcwd() + '/resources/login/login.html')

def checkpage(b):
    #num = b.get_element_count('xpath=//form[@id="myForm"]')
    b.get_text('xpath=//*[@class="infobar"]') == 'Logged In'

start_time = time.time()
for i in range(10000):
    b.get_element_count('xpath=//form[@id="myForm"]')
print('done')
print("--- %s seconds ---" % (time.time() - start_time))
b.close_browser()