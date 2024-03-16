from ATAG.src.atag.framework.environments.Data import DataLoad, DataSave, PathSave
import Observer
import EnvironmentControl
import numpy as np
import time

class BrowserControl(EnvironmentControl):
    def __init__(self, config):

        self.config = config

        self.load = DataLoad(self.config)
        self.save = DataSave(self.config)
        self.action_dim = self.load.lenActions()
        self.state_dim = self.load.lenElements()

        self.test_env = self.config.setup_env()
        self.observer = Observer(self, self.config)

        self.config.setup_test()
        self.prevObs = []

    def reset(self):
        self.config.teardown_test()
        self.config.setup_test()
        self.observer.reset()
        initial_obs = self.observer.observe()[0]
        self.prevObs = [initial_obs]
        return initial_obs

    def terminate(self):
        self.config.teardown_test()

    def take_action(self, act, args, kwargs):
        try:
            getattr(self.test_env, act)(*args, **kwargs)
            return self.config.env_parameters.get('passed_action_cost')
        except AssertionError:
            return self.config.env_parameters.get('failed_action_cost')

    def stagnation_reward(self, obs):
        if any(np.array_equal(obs, x) for x in self.prevObs):
            return self.config.env_parameters.get('stagnation_cost')
        return 0

    def get_selected_action(self, act):
        if not isinstance(act, int):
            act = act.argmax()
        return self.load.get_action(act)

    def step(self, act, evaluation=False):
        selected_act = self.get_selected_action(act)
        if evaluation: print(selected_act)
        act_reward = self.take_action(selected_act['keyword'], selected_act['args'], {})
        obs, obs_reward, done = self.observer.observe()

        # Calculate reward and set previous observation
        # Reward signal: cost of possible failure, reward from observation and cost from possible stagnation
        reward = act_reward + obs_reward + self.stagnation_reward(obs)
        self.prevObs.append(obs)

        return obs, reward, done, {}


class BrowserObserver(Observer):
    def __init__(self, browser_env, config):
        self.done = False
        self.browser_env = browser_env
        self.test_env = browser_env.test_env
        self.config = config

        self.load = DataLoad(self.config)
        self.save = DataSave(self.config)
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

        for element in scannedElements:
            element['attributes'] = [attr for attr in element['attributes'] if attr['key'] != 'class']

        if self.config.data_collection.get('collect_data'):
            self.save.saveElements(scannedElements)
            self.save.saveActions(scannedElements)
        
        return np.array([1 if e in scannedElements else 0 for e in elements])

    def __observeTargets(self):
        reward_sum, self.done = self.config.state_rewards()
        return reward_sum

    def observe(self):
        try:
            self.config.env_ready()
            obs = self.__observeElements()
            reward = self.__observeTargets()
            if self.config.data_collection.get('collect_path'):
                self.pathsave.save(obs, self.done, self.config.label)

            return np.array(obs), reward, self.done
        except AssertionError:
            print("error")
            time.sleep(4)
            return np.zeros(self.browser_env.state_dim), 0, True