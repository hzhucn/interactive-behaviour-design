#!/usr/bin/env python3

import argparse
import glob
import os
import pickle
import sys
import threading

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import global_variables

parser = argparse.ArgumentParser()
parser.add_argument('run_dir')
args = parser.parse_args()


def find_score(pkl_path):
    with open(pkl_path, 'rb') as f:
        rollout = pickle.load(f)
    global_variables.env_creation_lock = threading.Lock()
    env = rollout.final_env_state.env.unwrapped
    env.step(0)
    digits = [(b & 15) + (b >> 4) * 10 for b in env.ale.getRAM()[56:59]]
    val = 10000 * digits[0] + 100 * digits[1] + digits[2]
    return val


for traj_dir in sorted(glob.glob(os.path.join(args.run_dir, 'demonstrations', 'trajectory_*'))):
    traj_file = os.path.basename(traj_dir)
    traj_path = os.path.join(traj_dir, traj_file)
    if not os.path.exists(traj_path):
        continue
    print(traj_dir)
    with open(traj_path, 'r') as f:
        lines = f.readlines()
        lines = lines[0], lines[-2], lines[-1]
    pkls = [l.split(',')[1] + '.pkl'
            for l in lines]
    scores = []
    for pkl in pkls:
        scores.append(find_score(os.path.join(traj_dir, pkl)))
    final_score = scores[-1]
    if final_score == 0:
        final_score = scores[-2]
    print(scores[0], '->', final_score)
