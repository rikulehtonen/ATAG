from Browser import Browser
from Browser.utils.data_types import SupportedBrowsers
import time
import os
from Browser import AssertionOperator
import multiprocessing
import time


def checkpage(e):
    #num = b.get_element_count('xpath=//form[@id="myForm"]')
    check = 'footer-poweredbyico' in e
    #print(check)

b = Browser(timeout="0 s", retry_assertions_for="0 ms")
b.new_browser(headless=False, browser=SupportedBrowsers.chromium)
b.new_context(
    acceptDownloads=True,
    viewport={"width": 700, "height": 500}
)
b.new_page('file://' + os.getcwd() + '/resources/login/login.html')
#b.new_page('https://fi.wikipedia.org/')
time.sleep(2)
b.click('xpath=//button[@id="loginBox"]')

ids = """Array.prototype.map.call(document.getElementsByTagName('*'), (element) => 
{ 
    if (element.offsetParent === null)
    {
        return null 
    } 
    else 
    { 
        return {'tag': element.tagName, 'text': element.textContent, 'value': element.value, 'id': element.getAttribute('id') } 
    } 
}).filter(elements => { return elements !== null })"""

start_time = time.time()
elements = b.evaluate_javascript('xpath=//html', ids)

#for i in range(10000):
#    checkpage(elements)

print("--- %s seconds ---" % (time.time() - start_time))

print(elements)


#e = b.get_element_count('xpath=//*')
#print(e)

#start_time = time.time()
#ids = [b.get_property(elem,'id') for elem in e]

#print('done')
#print("--- %s seconds ---" % (time.time() - start_time))

#print(ids)
b.close_browser()