import numpy as np

class EnvironmentControl(object):
    def __init__(self, config):
        
        self.config = config
        self.observer = None
        self.previousObs = []

    def init_test(self):
        self.test_env = self.config.setup_env()
        self.config.setup_test()
        self.previousObs = []

    def reset(self):
        self.config.teardown_test()
        self.config.setup_test()
        self.observer.reset()
        initial_obs = self.observer.observe()[0]
        self.prevObs = [initial_obs]
        return initial_obs

    def terminate(self):
        self.config.teardown_test()

    def take_action(self, selected_act):
        act, args, kwargs = selected_act['keyword'], selected_act['args'], {}
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

    def step(self, act):
        selected_act = self.get_selected_action(act)
        action_reward = self.take_action(selected_act)
        obs, obs_reward, done = self.observer.observe()

        # Calculate reward and set previous observation
        # Reward signal: cost of possible failure, reward from observation and cost from possible stagnation
        reward = action_reward + obs_reward + self.stagnation_reward(obs)
        self.prevObs.append(obs)

        return obs, reward, done, {}
