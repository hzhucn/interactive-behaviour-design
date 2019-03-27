#!/usr/bin/env python3

import argparse
import glob
import os
import json
import pickle
import sys
import threading
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import global_variables

parser = argparse.ArgumentParser()
parser.add_argument('run_dir')
args = parser.parse_args()

def has_landed_successfully(pkl_path):
    with open(pkl_path, 'rb') as f:
        rollout = pickle.load(f)
    global_variables.env_creation_lock = threading.Lock()
    env = rollout.final_env_state.env.unwrapped
    observation = rollout.obses[-1]

    xPos = observation[0]
    isBetweenFlags = abs(xPos) < 0.1
    return has_landed(env, observation) and isBetweenFlags


def has_landed(env, observation):
    xPos, xVel, yVel, = observation[0], observation[2], observation[3]
    bothLegsOnGround = all([env.unwrapped.legs[i].ground_contact for i in range(2)])
    hasStopped = abs(xVel) < 0.1 and abs(yVel) < 0.1
    return bothLegsOnGround and hasStopped

def plot_successes(successes, success_rate, save_path): #scatter plot
    margin = 0.1

    plt.figure()
    plt.axis([0 - margin, len(successes) + 1, 0 - margin, 1 + margin])
    plt.xlabel("Demonstration #")
    plt.ylabel("Success (1)")
    plt.title("Successes over demonstrations")

    xs = list(range(len(successes)))
    scalars = [1 if x else 0 for x in successes]
    paths = plt.scatter(xs, scalars)

    #plt.show(block=True)
    plt.text(np.median(xs), 0.5, r'Success rate: {}%'.format(np.round(success_rate, 3)))
    plt.savefig(os.path.join(save_path, 'successes_over_time.png'))


successes = []
for traj_dir in sorted(glob.glob(os.path.join(args.run_dir, 'demonstrations', 'trajectory_*'))):
    traj_file = os.path.basename(traj_dir)
    traj_path = os.path.join(traj_dir, traj_file)
    if not os.path.exists(traj_path):
        continue
    print(traj_dir)
    with open(traj_path, 'r') as f:
        lines = f.readlines()

    last_line = lines[-1].split(',')
    chosen_hash = last_line[1]
    if chosen_hash == "equal":
        metadata_filename = last_line[0]
        metadata_path = os.path.join(traj_dir, metadata_filename)
        with open(metadata_path, 'r') as f:
            rollout_hashes = json.load(f)
            chosen_hash = rollout_hashes[0]

    last_pkl = chosen_hash + '.pkl'
    was_success = has_landed_successfully(os.path.join(traj_dir, last_pkl))
    successes.append(was_success)
    print("\t", was_success)

num_successes = len([x for x in successes if x])
success_rate = (num_successes / len(successes)) * 100
print("Oracle demonstration landing success rate: {}".format(success_rate))

plot_successes(successes, success_rate, os.path.join(args.run_dir, 'demonstrations'))