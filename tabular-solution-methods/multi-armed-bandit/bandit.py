import numpy as np
from gym import Env, spaces
from gym.spaces import Discrete
from gym.utils import seeding

class BanditEnv(Env):
    def __init__(self, n_bandits=10):
        self.n_bandits = n_bandits
        self._seed = self._seed()
        self.action_space = Discrete(n_bandits, seed = self._seed, start = 1) # actions are bandit indices
        self.observation_space = Discrete(1) # single state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed