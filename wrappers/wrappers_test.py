#!/usr/bin/env python

import multiprocessing
import os
import sys
import threading
import unittest

import numpy as np
from gym import Env

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import global_variables
from wrappers.dummy_env import DummyEnv
from wrappers.util_wrappers import SaveMidStateWrapper, StateBoundaryWrapper, RepeatActions



class TestSaveMidState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Needed to be able to restore env from state
        global_variables.env_creation_lock = threading.Lock()

    def test_distribution(self):
        """
        Confirm that distribution of steps from which states are saved is approximately uniform.
        """
        env = DummyEnv(max_steps=100)
        env = StateBoundaryWrapper(env)
        q = multiprocessing.Queue()
        np.random.seed(0)
        env = SaveMidStateWrapper(env, q, verbose=False)
        env.reset()
        states = []
        for _ in range(1000):
            done = False
            while not done:
                obs, _, done, _ = env.step(action=None)
            env.reset()
            states.append(q.get())

        state_steps = [state.step_n for state in states]
        # Assuming states saved at intervals of 10 steps
        assert env.SAVE_EVERY_NTH_STEP == 10
        counts, bins = np.histogram(state_steps, bins=np.arange(5, 115, 10))
        np.testing.assert_allclose(counts, 10 * [100], atol=20)

    def test_short_episodes(self):
        """
        Check that a state gets saved even for short episodes
        """
        q = multiprocessing.Queue()
        states = []
        for n_extra_steps in range(10):
            env = DummyEnv(max_steps=SaveMidStateWrapper.SAVE_EVERY_NTH_STEP - 5 + n_extra_steps)
            env = StateBoundaryWrapper(env)
            env = SaveMidStateWrapper(env, q, verbose=False)
            env.reset()
            done = False
            while not done:
                obs, _, done, _ = env.step(action=None)
            env.reset()
            states.append(q.get())
        self.assertTrue(all([state.env.step_n > 0 for state in states]))


class TestRepeatActions(unittest.TestCase):
    def test(self):
        class DummyEnv(Env):
            def __init__(self):
                Env.__init__(self)
                self.step_n = None
                self.actions = None

            def _obs(self):
                return self.step_n

            def reset(self):
                self.step_n = 0
                self.actions = []
                return self._obs()

            def step(self, action):
                self.actions.append(action)
                self.step_n += 1
                reward = action
                done = False
                info = None
                return self._obs(), reward, done, info

        env = DummyEnv()
        env = RepeatActions(env, repeat_n=1)
        env.reset()
        rewards = []
        for n in range(3):
            obs, reward, done, info = env.step(n)
            rewards.append(reward)
        assert env.unwrapped.actions == [0, 1, 2], env.unwrapped.actions
        assert rewards == [0, 1, 2], rewards

        env = DummyEnv()
        env = RepeatActions(env, repeat_n=2)
        env.reset()
        rewards = []
        for n in range(3):
            obs, reward, done, info = env.step(n)
            rewards.append(reward)
        assert env.unwrapped.actions == [0, 0, 1, 1, 2, 2], env.unwrapped.actions
        assert rewards == [0, 2, 4], rewards


if __name__ == '__main__':
    unittest.main()
