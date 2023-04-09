from Browser import Browser
from Browser.utils.data_types import SupportedBrowsers
import time
import os
from Browser import AssertionOperator
import multiprocessing
import time

from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Union


def checkpage(e):
    #num = b.get_element_count('xpath=//form[@id="myForm"]')
    e.get_element('xpath=//*[@id="loginBox"]')

b = Browser(timeout="0 s", retry_assertions_for="0 ms")
b.new_browser(headless=False, browser=SupportedBrowsers.chromium)
b.new_context(
    acceptDownloads=True,
    viewport={"width": 700, "height": 500}
)
#b.new_page('file://' + os.getcwd() + '/resources/login/login.html')
b.new_page('https://fi.wikipedia.org/')
time.sleep(2)



#e = b.get_element_count('xpath=//*')
#print(e)

start_time = time.time()
#ids = [b.get_property(elem,'id') for elem in e]

print('done')
print("--- %s seconds ---" % (time.time() - start_time))

#print(ids)
b.close_browser()