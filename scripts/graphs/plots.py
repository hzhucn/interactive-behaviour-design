#!/usr/bin/env python3

"""
Feed me a directory which has been organised by organise_runs.py (I'll look for e.g. Seaquest/DRLHP/2)
"""

import argparse
import fnmatch
import glob
import multiprocessing
import os
import re
import subprocess
import sys
import unittest
from collections import namedtuple

import dateutil
import matplotlib
import numpy as np
import scipy.stats
import tensorflow as tf
from matplotlib.pyplot import close, fill_between, xlim

matplotlib.use('Agg')

from pylab import plot, xlabel, ylabel, figure, legend, savefig, grid, ylim


# Event-reading utils

def find_files_matching_pattern(pattern, path):
    result = []
    for root, dirs, files in os.walk(path, followlinks=True):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def read_events_file(events_filename):
    events = {}
    try:
        for event in tf.train.summary_iterator(events_filename):
            for value in event.summary.value:
                if value.tag not in events:
                    events[value.tag] = []
                events[value.tag].append((event.wall_time, value.simple_value))
    except Exception as e:
        print(f"While reading '{events_filename}':", e)
    return events


def read_all_events(directory):
    events_files = find_files_matching_pattern('events.out.tfevents*', directory)
    pool = multiprocessing.Pool(processes=32)
    events_in_each_file = pool.map(read_events_file, events_files)
    all_events = {}
    for events in events_in_each_file:
        all_events.update(events)
    return all_events


# Value interpolation stuff

def interpolate_values(x_y_tuples, new_xs):
    xs, ys = zip(*x_y_tuples)
    if new_xs[-1] < xs[0]:
        raise Exception("New x values end before old x values begin")
    if new_xs[0] > xs[-1]:
        raise Exception("New x values start after old x values end")

    new_ys = np.interp(new_xs, xs, ys,
                       left=None, right=None)  # use None if we don't have data
    return new_ys


class TestInterpolateValues(unittest.TestCase):
    def test(self):
        timestamps = [0, 1, 2, 3]
        values = [0, 10, 20, 30]
        timestamps2 = [-1, 0, 0.5, 1, 1.1, 3, 3.1]
        interpolated_values = interpolate_values(list(zip(timestamps, values)), timestamps2)
        self.assertEqual(interpolated_values, [None, 0.0, 5.0, 10.0, 11.0, 30.0, None])


# Plotting stuff

def smooth_values(values, smoothing):
    smoothed_values = []
    last = values[0]
    for v in values:
        smoothed_values.append(smoothing * last + (1 - smoothing) * v)
        last = smoothed_values[-1]
    return smoothed_values


def interpolate_steps(timestamp_value_tuples, timestamp_step_tuples):
    step_timestamps, steps = zip(*timestamp_step_tuples)
    value_timestamps, values = zip(*timestamp_value_tuples)

    if len(timestamp_step_tuples) < len(timestamp_value_tuples):
        # Use step timestamps for interpolation
        steps = steps
        values = interpolate_values(timestamp_value_tuples, step_timestamps)
    elif len(timestamp_value_tuples) < len(timestamp_step_tuples):
        # Use value timestamps for interpolation
        steps = interpolate_values(timestamp_step_tuples, value_timestamps)
        values = values
    else:
        pass

    # interpolate_values uses None to signal "couldn't interpolate this value"
    # (because we didn't have data at the beginning or end); let's remove those points
    drop_idxs = []
    for i in range(len(steps)):
        if steps[i] is None:
            drop_idxs.append(i)
    steps = [steps[i] for i in range(len(steps)) if i not in drop_idxs]
    values = [values[i] for i in range(len(values)) if i not in drop_idxs]

    return steps, values


def find_training_start(run_dir):
    # The auto train script exits once training has started properly
    return os.path.getmtime(os.path.join(run_dir, 'auto_train.log'))


M = namedtuple('M', 'tag name smoothing fillsmoothing')


def detect_metrics(env_name, train_env_key):
    metrics = []
    if 'Lunar Lander' in env_name:
        metrics.append(M(f'{train_env_key}/reward_sum', 'Reward', 0.99, 0.99))
        metrics.append(M(f'{train_env_key}/crash_rate', 'Crash rate', 0.995, 0.99))
        metrics.append(M(f'{train_env_key}/successful_landing_rate', 'Successful landing rate', 0.995, 0.99))
    if 'Seaquest' in env_name:
        metrics.append(M(f'{train_env_key}/reward_sum', 'Reward', 0.9, 0.9))
        metrics.append(M(f'{train_env_key}/n_diver_pickups', 'Diver pickups per episode', 0.99, None))
    if 'Fetch' in env_name:
        metrics.append(M(f'{train_env_key}/reward_sum_post_wrappers', 'Reward', 0.95, 0.9))
        metrics.append(M(f'{train_env_key}/gripper_to_block_cumulative_distance', 'Distance from gripper to block', 0.99, 0.95))
        metrics.append(M(f'{train_env_key}/block_to_target_cumulative_distance', 'Distance from block to target', 0.99, 0.99))
        metrics.append(M(f'{train_env_key}/success_rate', 'Success rate', 0.95, 0.95))
    return metrics


def make_timestamps_relative_hours(events):
    timestamps, values = zip(*events)
    relative_timestamps = [t - timestamps[0] for t in timestamps]
    relative_timestamps_hours = [t / 3600 for t in relative_timestamps]
    values = list(values)  # otherwise it's a tuple
    return relative_timestamps_hours, values


def differentiate(timestamps, values, hop):
    ts, dvs = [], []
    last_t, last_v = timestamps[0], values[0]
    assert len(timestamps) == len(values)
    for i in range(hop, len(timestamps), hop):
        t, v = timestamps[i], values[i]
        ts.append(last_t)
        dvs.append((v - last_v) / (t - last_t))
        last_t, last_v = t, v
    return ts, dvs


def match_lengths(xs_list, ys_list):
    # Make all series end at the same x value by discarding values from the end
    max_x = min([xs[-1] for xs in xs_list])
    for n in range(len(xs_list)):
        xs = xs_list[n]
        idxs = np.argwhere(np.array(xs) > max_x)
        if idxs:
            n_to_drop = len(xs) - idxs[0][0]
            print("Dropping {} values ({}% of total)".format(n_to_drop, int(100 * n_to_drop / len(xs))))
            assert len(xs_list[n]) == len(ys_list[n])
            xs_list[n] = xs_list[n][:-n_to_drop]
            ys_list[n] = ys_list[n][:-n_to_drop]


def plot_averaged(xs_list, ys_list, smoothing, fillsmoothing, color, label):
    # Interpolate all data to have common x values
    all_xs = set([x for xs in xs_list for x in xs])
    all_xs = sorted(list(all_xs))
    for n in range(len(xs_list)):
        ys_list[n] = interpolate_values(list(zip(xs_list[n], ys_list[n])), all_xs)

    # interpolate_values uses None to signal "couldn't interpolate this value"
    # (because we didn't have data at the beginning or end); let's remove those points
    drop_idxs = []
    for ys in ys_list:
        for i in range(len(ys)):
            if ys[i] is None:
                drop_idxs.append(i)

    for n in range(len(ys_list)):
        ys_list[n] = [ys_list[n][i] for i in range(len(ys_list[n])) if i not in drop_idxs]
    all_xs = [all_xs[i] for i in range(len(all_xs)) if i not in drop_idxs]
    assert len(set([len(all_xs)] + [len(ys) for ys in ys_list])) == 1

    # Subsample /before/ smoothing so that we get the same level of smoothness no matter how dense the data
    plot_width_pixels = 1500  # determined by manually checking the figure
    # binned_statistic chooses equally-sized bins, and some of those bins might be empty.
    # For those bins, binned_statistic will returns NaN values.
    # Let's be careful about those.
    x_bin_centres = scipy.stats.binned_statistic(all_xs, all_xs, statistic='mean', bins=plot_width_pixels).statistic
    assert len(x_bin_centres.shape) == 1
    x_nan_idxs = np.argwhere(np.isnan(x_bin_centres))[:, 0]
    x_bin_centres = [v for i, v in enumerate(x_bin_centres) if i not in x_nan_idxs]
    ys_list_binned = []
    for ys in ys_list:
        binned_means = scipy.stats.binned_statistic(all_xs, ys, statistic='mean', bins=plot_width_pixels).statistic
        y_nan_idxs = np.argwhere(np.isnan(binned_means))[:, 0]
        assert np.array_equal(x_nan_idxs, y_nan_idxs)
        binned_means = [v for i, v in enumerate(binned_means) if i not in y_nan_idxs]
        ys_list_binned.append(binned_means)

    mean_ys = np.mean(ys_list_binned, axis=0)  # Average across seeds
    smoothed_mean_ys = smooth_values(mean_ys, smoothing=smoothing)

    plot(x_bin_centres, smoothed_mean_ys, color=color, label=label, alpha=0.9)
    if fillsmoothing is not None:
        std = np.std(ys_list_binned, axis=0)
        fill_between(x_bin_centres,
                     smooth_values(smoothed_mean_ys - std, fillsmoothing),
                     smooth_values(smoothed_mean_ys + std, fillsmoothing), color=color, alpha=0.2)

    grid(True)

    return np.min(smoothed_mean_ys), np.max(smoothed_mean_ys)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--runs_dir')
    group.add_argument('--test', action='store_true')
    parser.add_argument('--cols', type=int, default=4)
    parser.add_argument('--max_steps', type=float)
    parser.add_argument('--max_hours', type=float)
    parser.add_argument('--train_env_key', default='env')
    args = parser.parse_args()

    if args.test:
        sys.argv.pop(1)
        unittest.main()

    for f in glob.glob('*.png'):
        os.remove(f)

    # I'm looking for Seaquest/DRLHP/0
    for env_dir in os.scandir(args.runs_dir):
        escaped_env_name = env_dir.name.replace(' ', '_').lower()

        events_by_run_type_by_seed = {}
        for run_type_dir in os.scandir(env_dir.path):
            events_by_run_type_by_seed[run_type_dir.name] = {}
            for seed_dir in os.scandir(run_type_dir.path):
                print(f"Reading events for {env_dir.name}/{run_type_dir.name}/{seed_dir.name}")
                events = read_all_events(seed_dir.path)

                # Filter out events from the initial pretraining period
                # (in particular, for DRLHP, filter out env rewards from pretraining)
                try:
                    training_start_timestamp = find_training_start(seed_dir.path)
                    first_step = events['policy_master/n_total_steps'][0][1]
                except:
                    # No big deal - we're in RL-only or BC-only mode
                    pass
                else:
                    for tag in events:
                        events[tag] = [(t, v) for t, v in events[tag] if t >= training_start_timestamp]
                    events = {k: v for k, v in events.items() if v}
                    # Reset the steps to start from 0 after the pretraining period
                    events['policy_master/n_total_steps'] = [(t, step - first_step)
                                                             for t, step in events['policy_master/n_total_steps']]

                events_by_run_type_by_seed[run_type_dir.name][seed_dir.name] = events

        print("Plotting...")
        metrics = detect_metrics(env_dir.name, args.train_env_key)

        # Plot metrics by time
        for metric_n, metric in enumerate(metrics):
            figure(metric_n)
            escaped_metric_name = metric.name.replace(' ', '_').replace('.', '').lower()
            all_min_y = float('inf')
            all_max_y = -float('inf')
            for run_type_n, run_type in enumerate(events_by_run_type_by_seed):
                color = f"C{run_type_n}"
                xs_list = []
                ys_list = []
                for events in events_by_run_type_by_seed[run_type].values():
                    if metric.tag not in events:
                        print(f"Error: couldn't find metric '{metric.tag}' in run '{run_type}'", file=sys.stderr)
                        exit(1)
                    relative_timestamps_hours, values = make_timestamps_relative_hours(events[metric.tag])
                    if args.max_hours:
                        values = np.extract(np.array(relative_timestamps_hours) < args.max_hours,
                                            values)
                        relative_timestamps_hours = np.extract(np.array(relative_timestamps_hours) < args.max_hours,
                                                               relative_timestamps_hours)
                    xs_list.append(relative_timestamps_hours)
                    ys_list.append(values)
                min_y, max_y = plot_averaged(xs_list, ys_list, metric.smoothing, metric.fillsmoothing, color, run_type)
                if args.max_hours:
                    xlim([0, args.max_hours])
                all_max_y = max_y if max_y > all_max_y else all_max_y
                all_min_y = min_y if min_y < all_min_y else all_min_y
            xlabel("Hours")
            ylabel(metric.name)
            legend()
            ylim([all_min_y, all_max_y])
            fig_filename = '{}_{}_by_time.png'.format(escaped_env_name, escaped_metric_name)
            savefig(fig_filename, dpi=300, bbox_inches='tight')

        close('all')

        # Plot metrics by step
        for metric_n, metric in enumerate(metrics):
            figure(metric_n)
            escaped_metric_name = metric.name.replace(' ', '_').replace('.', '').lower()
            all_min_y = float('inf')
            all_max_y = -float('inf')
            for run_type_n, run_type in enumerate(events_by_run_type_by_seed):
                color = f"C{run_type_n}"
                xs_list = []
                ys_list = []
                for events in events_by_run_type_by_seed[run_type].values():
                    if 'policy_master/n_total_steps' not in events:
                        continue
                    steps, values = interpolate_steps(events[metric.tag], events['policy_master/n_total_steps'])
                    if args.max_steps:
                        values = np.extract(np.array(steps) < args.max_steps, values)
                        steps = np.extract(np.array(steps) < args.max_steps, steps)
                    xs_list.append(steps)
                    ys_list.append(values)
                if not xs_list:
                    continue
                min_y, max_y = plot_averaged(xs_list, ys_list, metric.smoothing, metric.fillsmoothing, color, run_type)
                if args.max_steps:
                    xlim([0, args.max_steps])
                all_max_y = max_y if max_y > all_max_y else all_max_y
                all_min_y = min_y if min_y < all_min_y else all_min_y
            legend()
            xlabel("Steps")
            ylabel(metric.name)
            ylim([all_min_y, all_max_y])
            fig_filename = '{}_{}_by_step.png'.format(escaped_env_name, escaped_metric_name)
            savefig(fig_filename, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
