from .observer import Observer
from .datahandler import DataLoad, DataSave
import numpy as np

class BrowserEnv:
    def __init__(self, config):

        self.config = config

        self.load = DataLoad(self.config)
        self.save = DataSave(self.config)
        self.action_dim = self.load.lenActions()
        self.state_dim = self.load.lenElements()

        self.test_env = self.config.setup_env()
        self.observer = Observer(self, self.config, self.load, self.save)
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
