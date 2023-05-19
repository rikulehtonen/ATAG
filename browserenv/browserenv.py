from .observer import Observer
from .datahandler import DataLoad, DataSave

class BrowserEnv:
    def __init__(self, config):

        self.config = config

        self.load = DataLoad(self.config)
        self.save = DataSave(self.config)
        self.action_dim = self.load.lenActions()
        self.state_dim = self.load.lenElements()

        self.test_env = self.config.setup_env()
        self.observer = Observer(self.test_env, self.config, self.load, self.save)
        self.config.setup_test()

    def reset(self):
        self.config.teardown_test()
        self.config.setup_test()
        self.observer.reset()
        return self.observer.observe()

    def terminate(self):
        self.config.teardown_test()

    def take_action(self, act, args, kwargs):
        try:
            getattr(self.test_env, act)(*args, **kwargs)
            return self.config.env_parameters.get('passed_action_cost')
        except AssertionError:
            return self.config.env_parameters.get('failed_action_cost')

    def get_selected_action(self, act):
        return self.load.get_action(act.argmax())

    def step(self, act):
        selected_act = self.get_selected_action(act)
        act_reward = self.take_action(selected_act['keyword'], selected_act['args'], {})
        obs, obs_reward, done = self.observer.observe()
        reward = act_reward + obs_reward

        return obs, reward, done
