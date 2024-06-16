import numpy as np
from gym import Env, spaces
from gym.spaces import Discrete, Box
from gym.utils import seeding

class BanditEnv(Env):
    metadata = {'render_modes': []}
    def __init__(self, n_bandits=10):
        self.n_bandits = n_bandits
        self._seed = self._seed()
        self.action_space = Discrete(n_bandits, seed = self._seed) # actions are bandit indices
        self.observation_space = Discrete(1) # single state
        self.Q = Box(low = 0, high = 1, shape = (n_bandits,), dtype = np.float32) # estimated action values
        self.N = Box(low = 0, high = 1, shape = (n_bandits,), dtype = np.float32) # action counts for each bandit

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def reset(self):
        pass

    def step(self, action):
        pass

    def update_value_estimates(self, action, reward):
        self.Q[action] += 1 / self.N[action] * (reward - self.Q[action])