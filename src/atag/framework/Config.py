from abc import ABC, abstractmethod

class Config(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def setup_env(self):
        pass
    
    @abstractmethod
    def teardown_env(self):
        pass

    @abstractmethod
    def setup_test(self):
        pass

    @abstractmethod
    def teardown_test(self):
        pass

    @abstractmethod
    def env_ready(self):
        pass
    
    @abstractmethod
    def state_rewards(self):
        pass
