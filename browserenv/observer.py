from .datahandler import DataLoad, DataSave
from .datahandler import PathSave
import time
import numpy as np
from Browser import AssertionOperator


class Observer:
    def __init__(self, test_env, config, load, save):
        self.done = False
        self.test_env = test_env
        self.config = config

        self.load = load
        self.save = save
        self.pathsave = PathSave(config)

    def reset(self):
        self.done = False
        self.pathsave.reset()

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
        scannedElements = self.test_env.evaluate_javascript('xpath=//html', ids)
        elements = self.load.elements

        if self.config.data_collection.get('collect_data'):
            self.save.saveElements(scannedElements)
            self.save.saveActions(scannedElements)
        
        return np.array([1 if e in scannedElements else 0 for e in elements])

    def __observeTargets(self):
        reward_sum, self.done = self.config.state_rewards()
        return reward_sum

    def observe(self):
        self.config.env_ready()
        obs = self.__observeElements()
        reward = self.__observeTargets()
        if self.config.data_collection.get('collect_path'):
            self.pathsave.save(obs, self.done)

        return np.array(obs), reward, self.done