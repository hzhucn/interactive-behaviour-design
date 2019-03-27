import copy
import pickle
import unittest

import cv2
import gym
import numpy as np
from gym.envs import register as gym_register
from gym.envs.box2d import LunarLander
from gym.envs.box2d.lunar_lander import VIEWPORT_W, SCALE, LEG_DOWN, FPS, VIEWPORT_H

from utils import TimerContext


class LunarLanderStateful(LunarLander):
    """
    A version of Lunar Lander that supports saving and restoring the complete game state
    by playing back actions since reset.
    """

    def __init__(self):
        self.actions = None
        self.debug = False
        LunarLander.__init__(self)

    def get_action_meanings(self):
        return ['NOOP', 'FIRELEFT', 'FIREMAIN', 'FIRERIGHT']

    def reset(self):
        self.np_random_at_reset = copy.deepcopy(self.np_random)
        obs = LunarLander.reset(self)
        self.actions = []
        return obs

    def step(self, action):
        if self.actions is not None:
            self.actions.append(action)
        return LunarLander.step(self, action)

    def get_state(self):
        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
        ]
        return state

    def render(self, mode='human'):
        if mode == 'human' or not self.debug:
            return LunarLander.render(self, mode)

        im = np.array(LunarLander.render(self, mode='rgb_array'))
        state = self.get_state()
        for n, (label, val) in enumerate([('distance', np.sqrt(state[0] ** 2 + state[1] ** 2)),
                                          ('velocity', np.sqrt(state[2] ** 2 + state[3] ** 2)),
                                          ('angle', abs(state[4]))]):
            cv2.putText(im,
                        '{}: {:.2f}'.format(label, val),
                        org=(5, 20 + n * 20),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=1.0,
                        color=(255, 255, 255),
                        thickness=1)
        return im

    def __getstate__(self):
        ezpickle_state = LunarLander.__getstate__(self)
        return ezpickle_state, self.np_random_at_reset, self.actions

    def __setstate__(self, state):
        ezpickle_state, np_random, actions = state
        LunarLander.__setstate__(self, ezpickle_state)
        self.np_random = np_random
        self.reset()
        for action in actions:
            self.step(action)


def register():
    gym_register(
        id='LunarLanderStateful-v0',
        entry_point=LunarLanderStateful,
        max_episode_steps=1000,
    )


class Test(unittest.TestCase):
    def test_state_restoration(self):
        # Test 1: starting with a fresh environment each time
        for _ in range(5):
            env = gym.make('LunarLanderStateful-v0')
            env.reset()
            self.clone_test(env)
        # Test 2: same environment, only reset
        env = gym.make('LunarLanderStateful-v0')
        for _ in range(5):
            env.reset()
            self.clone_test(env)

    def clone_test(self, env):
        for _ in range(30):
            env.step(env.action_space.sample())
        with TimerContext('Pickling and unpickling'):
            env2 = pickle.loads(pickle.dumps(env))
        self.assertEqual(env.unwrapped.lander.position.tuple,
                         env2.unwrapped.lander.position.tuple)


if __name__ == '__main__':
    register()
    unittest.main()
