from collections import deque

import gym
import numpy as np
from gym import Wrapper, ObservationWrapper
from gym.envs.robotics import FetchPickAndPlaceEnv
from gym.spaces import Discrete
from gym.wrappers import FlattenDictWrapper

from utils import RunningProportion
from wrappers.util_wrappers import SaveEpisodeStats


def decode_fetch_obs(obs):
    obs_by_name = {
        'grip_pos': obs[:3],
        'object_pos': obs[3:6],
        'object_rel_pos': obs[6:9],
        'gripper_state': obs[9:11],
        'object_rot': obs[11:14],
        'object_velp': obs[14:17],
        'object_velr': obs[17:20],
        'grip_velp': obs[20:23],
        'grip_vel': obs[23:25],
        'goal_pos': obs[25:28],
        'goal_rel_object': obs[25:28] - obs[3:6],
    }
    return obs_by_name


class FetchPickAndPlaceObsWrapper(ObservationWrapper):
    def __init__(self, env, include_grip_obs):
        ObservationWrapper.__init__(self, env)
        if include_grip_obs:
            n = 7
        else:
            n = 6
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(n,), dtype='float32')
        self.include_grip_obs = include_grip_obs

    def observation(self, obs):
        obs_dict = decode_fetch_obs(obs)
        final_obs = np.concatenate([obs_dict['object_rel_pos'],
                                    obs_dict['goal_rel_object']])
        if self.include_grip_obs:
            final_obs = np.concatenate([final_obs,
                                        # The gripper is supposed to be symmetrical.
                                        # But if the grippers are closed on the block and the arm is dragging the
                                        # block across the table, both grippers can slightly translate in the
                                        # MuJoCo simulation. So we need to look at both gripper positions to get the
                                        # actual width.
                                        [np.sum(obs_dict['gripper_state'])]])
            assert final_obs.shape == (7,)
        else:
            assert final_obs.shape == (6,)
        return final_obs


class FetchPickAndPlaceRewardWrapper(Wrapper):
    def __init__(self, env, reward_mode, binary_gripper, grip_close_bonus, grip_open_bonus, early_termination):
        Wrapper.__init__(self, env)
        self.reward_mode = reward_mode
        self.binary_gripper = binary_gripper
        self.grip_close_bonus = grip_close_bonus
        self.grip_open_bonus = grip_open_bonus
        self.early_termination = early_termination
        self.close_gripper = True
        self.last_grip_width = None
        self.last_obs_by_name = None

    @staticmethod
    def _r(d):
        r = d ** 2
        r += 0.1 * np.log(d + 1e-3)  # maxes out at -0.69
        return -0.7 - r  # maxes out at 0.0

    @classmethod
    def _reward_nondelta(cls, obs_by_name):
        d1 = np.linalg.norm(obs_by_name['object_rel_pos'])
        d2 = np.linalg.norm(obs_by_name['goal_rel_object'])
        reward = cls._r(d1) + cls._r(d2)
        return reward

    @staticmethod
    def _reward_delta1(obs_by_name, last_obs_by_name):
        if last_obs_by_name is None:
            reward = 0
        else:
            gripper_to_object_delta = (np.linalg.norm(obs_by_name['object_rel_pos']) -
                                       np.linalg.norm(last_obs_by_name['object_rel_pos']))
            # full speed in one direction -> reward = 0.5; full speed in two directions -> reward = 1.0
            reward = -15 * (gripper_to_object_delta)
        return reward

    @staticmethod
    def _reward_delta2(obs_by_name, last_obs_by_name):
        if last_obs_by_name is None:
            reward = 0
        else:
            gripper_to_object_delta = (np.linalg.norm(obs_by_name['object_rel_pos']) -
                                       np.linalg.norm(last_obs_by_name['object_rel_pos']))
            object_to_goal_delta = (np.linalg.norm(obs_by_name['goal_pos'] - obs_by_name['object_pos']) -
                                    np.linalg.norm(last_obs_by_name['goal_pos'] - last_obs_by_name['object_pos']))
            # full speed in one direction -> reward = 0.5; full speed in two directions -> reward = 1.0
            reward = -15 * (gripper_to_object_delta + object_to_goal_delta)
        return reward

    def object_between_grippers(self, obs_by_name):
        return all([abs(d) < 0.02 for d in obs_by_name['object_rel_pos']])

    def step(self, action, return_decoded_obs=False):
        if self.binary_gripper:
            action = np.copy(action)
            if action[3] <= -0.001:
                self.close_gripper = True
            elif action[3] >= 0.001:
                self.close_gripper = False

            if self.close_gripper:
                # Close slowly at first, but when we get to about the width of the block,
                # grip much harder (so that we do actually hold the block)
                action[3] = -0.002
                if self.last_obs_by_name is not None:
                    last_grip_width = np.sum(self.last_obs_by_name['gripper_state'])
                    if last_grip_width < 0.055:
                        action[3] = -1
            else:
                action[3] = +0.002

        obs, reward_orig, done, info = self.env.step(action)
        obs_by_name = decode_fetch_obs(obs)

        if self.reward_mode == 'nondelta':
            reward = self._reward_nondelta(obs_by_name)
        elif self.reward_mode == 'delta1':
            reward = self._reward_delta1(obs_by_name, self.last_obs_by_name)
        elif self.reward_mode == 'delta2':
            reward = self._reward_delta2(obs_by_name, self.last_obs_by_name)
        else:
            raise RuntimeError(f"reward mode is '{self.reward_mode}'")

        # Considerations:
        # - We could give rewards based on gripper width. But that would add a bias to rewards which might mask the
        #   bonuses for e.g. moving in the right direction/ending the episode early - and that would confuse the
        #   oracle demonstrator. So we give rewards based on /changes/ in gripper width.
        # - The reward should be reasonably large (e.g. +2 instead of +1) so that the oracle demonstrator always
        #   closes the gripper before trying to move the block.
        if self.last_obs_by_name is not None:
            last_grip_width = np.sum(self.last_obs_by_name['gripper_state'])
            grip_width = np.sum(obs_by_name['gripper_state'])
            if self.grip_open_bonus and not self.object_between_grippers(obs_by_name):
                # ~ +2.0 reward for fully closed to fully open, spread out over a few steps
                reward += 20 * (grip_width - last_grip_width)
            if self.grip_close_bonus:
                if self.object_between_grippers(obs_by_name):
                    # ~ +2.0 reward for fully open to closed around block, spread out over a few steps
                    reward += 40 * (last_grip_width - grip_width)
                elif self.object_between_grippers(self.last_obs_by_name):
                    # Penalise moving away from block
                    # (Value must be chosen carefully so that agent can't reward hack
                    #  by partially closing on block, moving away from block, opening
                    #  grippers, and repeating)
                    reward += -3

        if self.early_termination and np.linalg.norm(obs_by_name['goal_rel_object']) < 0.03:
            reward += 1
            done = True

        self.last_obs_by_name = obs_by_name

        if return_decoded_obs:
            obs = obs_by_name

        return obs, reward, done, info

    def reset(self):
        self.last_obs_by_name = None
        return self.env.reset()


class FetchStatsWrapper(SaveEpisodeStats):
    def __init__(self, env):
        assert isinstance(env, FlattenDictWrapper)
        assert env.observation_space.shape == (28,)
        SaveEpisodeStats.__init__(self, env)
        self.stats = {}
        self.last_stats = {}
        self.aligned_proportion = None
        self.gripping_proportion = None
        self.successes = deque(maxlen=25)  # for 5 initial positions, 5 samples of each position
        self.last_obs = None

    def reset(self):
        if self.last_obs is not None:
            # We do this here rather than in step because we apply FetchStatsWrapper as part of the registered
            # environment and therefore before TimeLimit, so we never see done in step
            obs_by_name = decode_fetch_obs(self.last_obs)
            block_to_target_dist = np.linalg.norm(obs_by_name['goal_rel_object'])
            if block_to_target_dist < 0.05:
                self.successes.append(True)
            else:
                self.successes.append(False)
            self.stats['success'] = float(self.successes[-1])
            if len(self.successes) == self.successes.maxlen:
                self.stats['success_rate'] = self.successes.count(True) / len(self.successes)

        self.last_stats = dict(self.stats)
        self.stats['gripper_to_block_cumulative_distance'] = 0
        self.stats['block_to_target_cumulative_distance'] = 0
        self.stats['ep_frac_aligned_with_block'] = 0
        self.stats['ep_frac_gripping_block'] = 0
        self.stats['block_to_target_min_distance'] = float('inf')
        self.aligned_proportion = RunningProportion()
        self.gripping_proportion = RunningProportion()

        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.last_obs = obs

        obs_by_name = decode_fetch_obs(obs)
        self.stats['gripper_to_block_cumulative_distance'] += np.linalg.norm(obs_by_name['object_rel_pos'])
        b2t_dist = np.linalg.norm(obs_by_name['goal_rel_object'])
        self.stats['block_to_target_cumulative_distance'] += b2t_dist
        if b2t_dist < self.stats['block_to_target_min_distance']:
            self.stats['block_to_target_min_distance'] = b2t_dist

        aligned_with_block = (np.linalg.norm(obs_by_name['object_rel_pos']) < 0.04)
        self.aligned_proportion.update(float(aligned_with_block))
        self.stats['ep_frac_aligned_with_block'] = self.aligned_proportion.v

        grippers_closed = np.sum(obs_by_name['gripper_state']) < 0.05
        gripping_block = aligned_with_block and grippers_closed
        self.gripping_proportion.update(float(gripping_block))
        self.stats['ep_frac_gripping_block']  = self.gripping_proportion.v

        return obs, reward, done, info


# Why implement these as wrappers (having to invoke FetchEnv-private methods), rather than implement these as part of
# the environment itself? Because we want the changes to survive pickling; and since FetchPickAndPlaceEnv uses EzPickle,
# there's no easy way to persist e.g. the current initial block position index.

class FixedGoal(Wrapper):

    def __init__(self, env, fixed_goal):
        assert isinstance(env, FetchPickAndPlaceEnv)
        super(FixedGoal, self).__init__(env)
        self.fixed_goal = fixed_goal

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if self.fixed_goal:
            self.env.unwrapped.goal = np.array([1.2, 0.6, 0.6])
            obs = self.env.unwrapped._get_obs()
        return obs


class FixedBlockInitialPositions(Wrapper):
    all_initial_block_positions = [
        np.array([1.46192226, 0.9513729]),
        np.array([1.18499211, 0.94554907]),
        np.array([1.15782761, 0.45436904]),
        np.array([1.4359836, 0.50284814]),
        np.array([1.34399605, 0.69795044])
    ]

    def __init__(self, env, n_initial_block_positions):
        assert isinstance(env, FixedGoal)
        super(FixedBlockInitialPositions, self).__init__(env)
        if n_initial_block_positions is not None:
            self.initial_block_positions = deque(self.all_initial_block_positions[:n_initial_block_positions])
        else:
            self.initial_block_positions = None

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        obs = self.env.reset()
        if self.initial_block_positions is not None:
            # copied from _reset_sim in FetchEnv
            object_xpos = self.initial_block_positions[0]
            self.initial_block_positions.rotate(1)
            object_qpos = self.env.unwrapped.sim.data.get_joint_qpos('object0:joint')
            object_qpos[:2] = object_xpos
            self.env.unwrapped.sim.data.set_joint_qpos('object0:joint', object_qpos)
            self.env.unwrapped.sim.forward()
            obs = self.env.unwrapped._get_obs()
        return obs


class RandomInitialPosition(Wrapper):

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        cur_pos = self.env.unwrapped.sim.data.get_site_xpos('robot0:grip')
        gripper_target = cur_pos + np.random.uniform(low=-0.2, high=0.2, size=cur_pos.shape)
        gripper_target[2] = np.clip(gripper_target[2], 0.5, float('inf'))
        self.env.unwrapped.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        for _ in range(10):
            self.env.unwrapped.sim.step()
        obs, _, _, _ = self.env.step([0, 0, 0, 0])
        return obs
