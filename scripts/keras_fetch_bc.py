import argparse
import multiprocessing
import os
import sys
import time

import gym
import numpy as np
from gym.wrappers import Monitor
from keras import Sequential
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.engine.saving import load_model
from keras.layers import Dense
from keras.optimizers import Adam

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from wrappers.fetch_pick_and_place import RandomInitialPosition
from wrappers.fetch_pick_and_place_register import register
from wrappers.util_wrappers import LogEpisodeStats
from wrappers.wrappers_debug import DrawActions, DrawRewards


def get_action_smooth(obs):
    assert obs.shape == (7,)
    gripper_to_block = obs[:3]
    block_to_target = obs[3:6]
    gripper_width = obs[6]
    if (np.linalg.norm(gripper_to_block) > 0.01 and
            np.linalg.norm(block_to_target) > 0.1):  # Don't open grippers if blocking is slipping near target
        if np.linalg.norm(gripper_to_block[:2]) > 0.05:
            z = 0
        else:
            z = gripper_to_block[2]
        action = np.concatenate([3 * gripper_to_block[:2], [3 * z, 1]])
    elif gripper_width > 0.05:
        action = np.array([0, 0, 0, -1])
    else:
        action = np.concatenate([3 * block_to_target, [-1]])
    return action


def get_action_jerky(obs):
    assert obs.shape == (7,)
    gripper_to_block = obs[:3]
    block_to_target = obs[3:6]
    gripper_width = obs[6]
    if (np.linalg.norm(gripper_to_block) > 0.03 and
            np.linalg.norm(block_to_target) > 0.1):  # Don't open grippers if blocking is slipping near target
        di = np.argmax(np.abs(gripper_to_block))
        action = np.array([0., 0., 0., 1.])
        action[di] = 0.15 * np.sign(gripper_to_block[di])
    elif gripper_width > 0.05:
        action = np.array([0, 0, 0, -1])
    else:
        di = np.argmax(np.abs(block_to_target))
        action = np.array([0., 0., 0., -1.])
        action[di] = 0.15 * np.sign(block_to_target[di])
    return action


def gen_demonstrations(env_id, log_dir, n_demonstrations):
    register()
    env = gym.make(env_id)
    env = RandomInitialPosition(env)
    env = LogEpisodeStats(env, log_dir, '_demo')
    # env = DrawObses(env)
    # env = DrawActions(env)
    # env = Monitor(env, video_callable=lambda n: True, directory=log_dir, uid=111)

    obses, actions = [], []
    for n in range(n_demonstrations):
        print(f"Generating demonstration {n}...")
        obs, done = env.reset(), False
        while not done:
            action = get_action_smooth(obs)
            obses.append(obs)
            actions.append(action)
            obs, reward, done, info = env.step(action)

    return obses, actions


def loss(y_true, y_pred):
    # TODO check
    return K.mean(K.sum(K.square(y_true - y_pred), axis=-1), axis=0)


def run_test_env(env_id, log_dir, model_path, model_lock):
    register()
    test_env = gym.make(env_id)
    test_env = DrawActions(test_env)
    test_env = DrawRewards(test_env)
    test_env = LogEpisodeStats(test_env, log_dir, '_test')
    test_env = Monitor(test_env, video_callable=lambda n: n % 7 == 0, directory=log_dir, uid=999)

    n = 0
    while True:
        model_lock.acquire()
        model = load_model(model_path, custom_objects={'loss': loss})
        model_lock.release()
        obs, done = test_env.reset(), False
        while not done:
            a = model.predict(np.array([obs]))[0]
            obs, reward, done, info = test_env.step(a)
        sys.stdout.flush()
        sys.stderr.flush()
        print(f"Test episode {n} done!")
        n += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()

    model = Sequential()
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=4, activation='tanh'))
    model.compile(loss=loss, optimizer=Adam(lr=1e-3))

    env_id = 'FetchPickAndPlace-Repeat1-ContGripper-WithGripObs-5InitialBlockPos-FixedGoal-NonDelta-NoGripperBonus-v0'
    obses, actions = gen_demonstrations(env_id, os.path.join(args.log_dir, 'demos'), 100)
    obses = np.array(obses)
    actions = np.array(actions)

    ctx = multiprocessing.get_context('spawn')
    model_lock = ctx.Lock()
    model_path = os.path.join(args.log_dir, 'model.h5')
    test_log_dir = os.path.join(args.log_dir, 'test_env')
    test_eps_proc = ctx.Process(target=run_test_env, args=(env_id, test_log_dir, model_path, model_lock))
    model.fit(obses, actions, epochs=1)
    model.save(model_path)
    test_eps_proc.start()

    callback = TensorBoard(log_dir=args.log_dir)
    while True:
        model.fit(obses, actions, epochs=10, callbacks=[callback])
        model_lock.acquire()
        model.save(model_path)
        model_lock.release()


if __name__ == '__main__':
    main()
