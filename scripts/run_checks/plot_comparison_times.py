#!/usr/bin/env python3

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

def get_deltas(verbose=False):
    """
    :return: The deltas between the timestamps of rollout decisions
    """
    demonstrations_dir = args.demonstrations_dir[0]
    trajectory_prefix = "trajectory_"
    trajectory_dirs = [dir for dir in os.listdir(demonstrations_dir) if trajectory_prefix in dir]

    timestamp_deltas = []
    for td in trajectory_dirs:
        trajectory_dirname = os.path.join(demonstrations_dir, td)
        trajectory_filename = os.path.join(trajectory_dirname, td)

        if os.path.exists(trajectory_filename):
            with open(trajectory_filename, 'r') as f:
                lines = [x.strip() for x in f.readlines()]

            timestamps = []
            for line in lines:
                group_serial, chosen_rollout_hash, timestamp = line.split(',')
                timestamps += [timestamp]

            for i in range(len(timestamps) - 1):
                t1 = timestamps[i + 1]
                t2 = timestamps[i]
                delta = float(t1) - float(t2)
                timestamp_deltas += [delta]
        else:
            if verbose:
                print("Path: {} does not exist".format(trajectory_filename))

    return timestamp_deltas

def clean_data(timestamp_deltas, threshold=30):
    """
    :param timestamp_deltas: in seconds
    :param threshold: in seconds
    :return: removes any values in timestamp_deltas above threshold
    """
    return [td for td in timestamp_deltas if td < threshold]

def plot_deltas(timestamp_deltas):
    plt.figure()
    plt.xlabel("Time (s)")
    plt.ylabel("Counts")
    plt.title("Time taken to chose b/t rollouts")

    max_delta = int(round(np.max(timestamp_deltas))) + 5
    bins = list(range(max_delta))
    mean = round(np.mean(timestamp_deltas), 3)

    n, bins, patches = plt.hist(timestamp_deltas, bins)
    plt.text(mean, np.max(n), r'$\mu={}s$'.format(mean))
    plt.show(block=True)

parser = argparse.ArgumentParser()
parser.add_argument('demonstrations_dir', nargs='*') #this should be the dir that contains all the trajectory dirs
args = parser.parse_args()

timestamp_deltas = clean_data(get_deltas())
plot_deltas(timestamp_deltas)