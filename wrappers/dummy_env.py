from gym import Env
from gym.envs.registration import EnvSpec


class DummyEnv(Env):
    def __init__(self, max_steps):
        self.step_n = None
        self.max_steps = max_steps
        self.spec = EnvSpec('DummyEnv-v0')

    def step(self, action):
        obs = self.step_n
        reward = 0
        info = None
        if self.step_n < self.max_steps:
            done = False
        else:
            done = True
        self.step_n += 1
        return obs, reward, done, info

    def reset(self):
        self.step_n = 0