from .datahandler import DataLoad, DataSave
import time
import numpy as np
from Browser import AssertionOperator


class Observer:
    def __init__(self, browser, collectData, load, save):
        self.obsCount = 0
        self.obsLimit = 7
        self.done = False
        self.browser = browser
        self.collectData = collectData

        self.load = load
        self.save = save

    def reset(self):
        self.obsCount = 0
        self.done = False

    def __observeElements(self):
        ids = """Array.prototype.map.call(document.getElementsByTagName('*'), (element) => 
        { 
            if (element.offsetParent === null)
            {
                return null 
            } 
            else 
            { 
                return {'tag': element.tagName, 'text': element.textContent, 'value': element.value, 'attributes': Array.prototype.map.call(element.attributes, (e) => ({ 'key': e.nodeName, 'value': e.nodeValue }) ) }
            } 
        }).filter(elements => { return elements !== null })"""
        scannedElements = self.browser.evaluate_javascript('xpath=//html', ids)
        elements = self.load.elements

        if self.collectData:
            self.save.saveElements(scannedElements)
            self.save.saveActions(scannedElements)
        
        return np.array([1 if e in scannedElements else 0 for e in elements])

    def __parse_keyword(self,target):
        keyword, args = target['keyword'], target['args'].copy()
        if "AssertionOperator" in args[1]:
            args[1] = getattr(AssertionOperator, args[1].split('.')[1])
        return keyword, args

    def __observeTargets(self):
        reward_sum = [0] 
        for target in self.load.targets:
            try:
                keyword, args = self.__parse_keyword(target)
                getattr(self.browser, keyword)(*args, **{})
                reward_sum.append(target.get('positive_reward'))
                self.done = target.get('is_done')
            except AssertionError:
                reward_sum.append(target.get('negative_reward'))

        reward_sum = sum(filter(None, reward_sum))
        return reward_sum

    def __checkDone(self):
        if self.obsCount >= self.obsLimit:
            self.done = True

    def __checkReady(self):
        time.sleep(0.05)

    def observe(self):
        self.__checkReady()
        obs = self.__observeElements()
        reward = self.__observeTargets()
        self.__checkDone()
        self.obsCount += 1

        return obs, reward, self.done