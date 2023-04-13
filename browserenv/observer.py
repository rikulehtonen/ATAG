from .datahandler import DataLoad, DataSave
import time
import numpy as np

"""
    def observe(self):
        done = False
        reward = None
        obs = np.zeros(self.state_dim())

        for i in range(self.state_dim()):
            try:
                getattr(self.b, self.elements[i][0])(*self.elements[i][1], **{})
                obs[i] = 1.0
            except:
                obs[i] = 0.0

        for id in range(len(self.targets)):
            try:
                getattr(self.b, self.targets[id][0])(*self.targets[id][1], **{})
                reward = self.targets[id][2]
                done = self.targets[id][4]
                if done: break
            except:
                reward = self.targets[id][3]

        if self.steps > 5:
            done = True

        return obs, reward, done
"""
        #self.targets = [['get_element_states', ['xpath=//div[@id="logininfo"]', AssertionOperator.contains, 'visible'], 2.0, None, True]]

        #self.elements = [['get_element_states', ['xpath=//form[@id="myForm"]', AssertionOperator.contains, 'visible']],
        #                 ['get_element_states', ['xpath=//div[@id="loginFailed"]', AssertionOperator.contains, 'visible']],
        #                 ['get_element_states', ['xpath=//div[@id="logininfo"]', AssertionOperator.contains, 'visible']],
        #                 ['get_text', ['xpath=//input[@name="uname"]', AssertionOperator.equal, 'testaaja']],
        #                 ['get_text', ['xpath=//input[@name="psw"]', AssertionOperator.equal, 'testi']]]

        #self.actions = [['click', ['xpath=//button[@id="loginBox"]']],
        #                ['click', ['xpath=//button[@type="submit"]']],
        #                ['button', ['xpath=//button[@class="cancelbtn"]']],
        #                ['type_text', ['xpath=//input[@name="uname"]','testaaja']],
        #                ['type_text', ['xpath=//input[@name="psw"]','testi']]]


class Observer:
    def __init__(self, browser, collectData, load, save):
        self.obsCount = 0
        self.obsLimit = 6
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
                        return {'tag': element.tagName, 'text': element.textContent, 'id': element.getAttribute('id') } 
                    } 
                }).filter(elements => { return elements !== null })"""
        scannedElements = self.browser.evaluate_javascript('xpath=//html', ids)
        elements = self.load.elements
        #if self.collectData:
        #    self.save.elements(elements)
        return np.array([1 if e in scannedElements else 0 for e in elements])


    def __observeTargets(self):
        for target in self.load.targets:
            try:
                getattr(self.b, target['keyword'])(*target['args'], **{})
                return target['positive_reward']
            except:
                return target['negative_reward']


    def __checkDone(self):
        if self.obsCount >= self.obsLimit:
            self.done = True


    def __checkReady(self):
        time.sleep(2)


    def observe(self):
        self.__checkReady()
        obs = self.__observeElements()
        reward = self.__observeTargets()
        self.__checkDone()
        self.obsCount += 1

        return obs, reward, self.done