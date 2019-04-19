#!/usr/bin/env python

"""
Start a single training run, opening demonstrations in Chrome.
"""

import argparse
import os
import json
import random
import subprocess
import sys
import time
import uuid

import numpy as np
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils import save_args, get_git_rev


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_id')
    parser.add_argument('training_mode', choices=['reward_only', 'bc_only', 'reward_plus_bc'])
    parser.add_argument('segment_generation', choices=['demonstrations', 'drlhp', 'both'])
    parser.add_argument('run_name')
    parser.add_argument('--n_envs', type=int, default=16)
    parser.add_argument('--n_prefs_before_training', type=int, default=10)
    parser.add_argument('--n_initial_prefs', type=int, default=200)
    parser.add_argument('--n_demos_before_training', type=int, default=2)
    parser.add_argument('--n_initial_demos', type=int, default=7)
    parser.add_argument('--pretrain_reward_predictor_seconds', type=int, default=30)
    parser.add_argument('--tmux_sess')
    if os.path.exists('/efs'):
        default_log_dir = '/efs'
    else:
        default_log_dir = 'runs'
    parser.add_argument('--log_dir', default=default_log_dir)
    parser.add_argument('--n_seeds', type=int, default=1) #if n>1, use n *random seeds
    parser.add_argument('--seeds', nargs='*', type=int, default=[76,233,429])
    parser.add_argument('--disable_redo', action='store_true')
    parser.add_argument('--extra_args')
    parser.add_argument('--time', default=str(int(time.time())))
    parser.add_argument('--min_label_interval_seconds', type=int, default=3)
    parser.add_argument('--max_interactions', type=int, default=None)
    args = parser.parse_args()

    args.run_name += "_" + args.time

    git_rev = get_git_rev()
    log_dir = os.path.join(args.log_dir, f'{args.run_name}_{git_rev}')
    log_dir = os.path.abspath(log_dir)
    args.log_dir = log_dir

    return args


def main():
    #stagger = random.randint(1, 20)
    #time.sleep(stagger)
    args = get_args()
    os.makedirs(args.log_dir, exist_ok=True)

    if not args.tmux_sess:
        script_name = sys.argv[0]
        script_args = '"' + '" "'.join(sys.argv[1:]) + '"'
        cmd = f'python -u {script_name} {script_args} --tmux_sess {args.run_name} --time {args.time}'
        cmd += f' 2>&1 | tee {args.log_dir}/startup_demos.log'
        start_tmux_sess_with_cmd(sess_name=args.run_name, cmd=cmd)
        return

    save_args(args, args.log_dir, 'startup_demos_args.txt')

    seeds = args.seeds
    if args.n_seeds > 1:
        seeds = np.random.randint(0, 500, args.n_seeds)

    i = 0
    for seed in seeds:
        port = 5000 + i #get_open_port()
        base_url = f'http://localhost:{port}'

        log_dir = os.path.join(args.log_dir, f"{args.segment_generation}-{seed}")
        start_app(base_url, args.env_id, args.n_envs, port, seed, log_dir, args.tmux_sess, args.disable_redo,
                  args.extra_args)

        #ngrok_url = forward_port(port, args.tmux_sess, seed)
        add_master_policy(base_url)
        if args.segment_generation == 'drlhp':
            wait_for_drlhp_segments(base_url)
            print(f"Give comparisons at {base_url + '/compare_segments'}, can stop after {args.n_initial_prefs} prefs")
            wait_for_n_preferences(base_url, args.n_prefs_before_training)  # wait for # prefs => start training
            if args.training_mode in ['reward_only', 'reward_plus_bc']:
                print("Hold off on demonstrations")
                start_reward_predictor_training(base_url, args.pretrain_reward_predictor_seconds)
            start_training(base_url, args.training_mode, args.segment_generation)
            print("Start demonstrations again")
            wait_for_n_preferences(base_url, args.n_initial_prefs, args.n_prefs_before_training)
        elif args.segment_generation == 'demonstrations':
            wait_for_demonstration_rollouts(base_url) #start rollouts
            print(f"Rollouts started at {base_url + '/demonstrate'}, can stop after {args.n_initial_demos} demos")
            wait_for_n_demonstrations(base_url, args.n_demos_before_training) #wait for # demos => start training
            if args.training_mode in ['reward_only', 'reward_plus_bc']:
                print("Hold off on demonstrations")
                start_reward_predictor_training(base_url, args.pretrain_reward_predictor_seconds)
            start_training(base_url, args.training_mode, args.segment_generation)
            print("Start demonstrations again")
            wait_for_n_demonstrations(base_url, args.n_initial_demos, args.n_demos_before_training)
        elif args.segment_generation == 'both':
            # TODO
            wait_for_demonstration_rollouts(base_url)
            wait_for_drlhp_segments(base_url)
            wait_for_n_demonstrations(base_url, args.n_initial_demos)
            start_reward_predictor_training(base_url, args.pretrain_reward_predictor_seconds)
        else:
            raise Exception()
        if args.segment_generation == 'demonstrations':
            configure_env_resets(base_url)
        i += 1

def start_app(base_url, env_id, n_envs, port, seed, log_dir, tmux_sess, disable_redo, extra_args):
    cmd = f'python -u run.py {env_id} --n_envs {n_envs} --port {port} --render_segments --log_dir {log_dir} --seed {seed}'
    if not disable_redo:
        cmd += ' --redo_policy'
    if 'Seaquest' in env_id:
        cmd += ' --load_policy_ckpt_dir subpolicies/seaquest'
    elif 'LunarLander' in env_id:
        cmd += ' --load_policy_ckpt_dir subpolicies/lunarlander'
    elif 'Fetch' in env_id:
        cmd += ' --add_manual_fetch_policies'
    if extra_args is not None:
        cmd += ' ' + extra_args
    cmd += f' 2>&1 | tee {log_dir}/output.log'
    run_in_tmux_sess(tmux_sess, cmd, f"seed{seed}")
    print(f"Waiting for run {env_id} seed {seed} to start...")
    while True:
        try:
            requests.get(base_url + '/get_status')
        except:
            time.sleep(0.5)
        else:
            break

def add_master_policy(base_url):
    print("Adding master policy...")
    requests.get(base_url + '/run_cmd?cmd=add_policy&name=master').raise_for_status()
    while True:
        time.sleep(0.5)
        response = requests.get(base_url + '/get_status').json()
        if 'master' in response['Policies']:
            break


def wait_for_demonstration_rollouts(base_url):
    requests.get(base_url + '/generate_rollouts?policies=').raise_for_status()
    print("Waiting for demonstration rollouts...")
    check_url = base_url + '/get_rollouts'
    while True:
        response = requests.get(check_url).json()
        if response:
            break
        else:
            time.sleep(0.5)

def wait_for_drlhp_segments(base_url):
    requests.get(base_url + '/run_cmd?cmd=use_policy&name=master').raise_for_status()
    requests.get(base_url + '/run_cmd?cmd=training_mode&mode=no_training').raise_for_status()
    print("Waiting for segments...")
    check_url = base_url + '/get_comparison'
    while True:
        response = requests.get(check_url).json()
        if response:
            break
        else:
            time.sleep(0.5)

def wait_for_n_interactions(log_dir, n):
    search_string = f"Simulated {n} interactions"
    while True:
        try:
            oracle_log = open(os.path.join(log_dir, 'oracle.log'), 'r').read()
        except Exception as e:
            print("While reading oracle log:", e)
        else:
            if search_string in oracle_log:
                return True
        time.sleep(1)


def wait_for_n_demonstrations(base_url, n, curr_n_demos=0):
    last_n_demos = -1
    while curr_n_demos < n:
        if curr_n_demos > last_n_demos:
            print(f"Waiting for {n} demonstrations ({curr_n_demos} demos so far)")
            last_n_demos = curr_n_demos
        time.sleep(1)
        curr_n_demos = get_n_demos(base_url)

def wait_for_n_preferences(base_url, n, curr_n_prefs=0):
    last_n_prefs = -1
    print_freq = 10
    while curr_n_prefs < n:
        if curr_n_prefs > last_n_prefs:
            print(f"Waiting for {n} preferences (>= {curr_n_prefs} so far)")
            last_n_prefs = curr_n_prefs * print_freq
        time.sleep(1)
        curr_n_prefs = get_n_prefs(base_url)


def start_reward_predictor_training(base_url, seconds):
    print(f"Pretraining reward predictor for {seconds / 60} min...")
    requests.get(base_url + '/run_cmd?cmd=start_drlhp_training').raise_for_status()
    time.sleep(seconds)

def start_training(base_url, training_mode, segment_generation):
    print("Starting training...")
    requests.get(base_url + '/run_cmd?cmd=use_policy&name=master').raise_for_status()
    requests.get(base_url + f'/run_cmd?cmd=training_mode&mode={training_mode}')
    if training_mode in ['reward_only', 'reward_plus_bc']:
        requests.get(base_url + '/run_cmd?cmd=set_reward_source&src=drlhp').raise_for_status()

def configure_env_resets(base_url):
    requests.get(base_url + '/run_cmd?cmd=add_reset_pool&name=random_states_from_episode&max_len=100')
    requests.get(base_url + '/run_cmd?cmd=use_reset_pool&from=training&name=random_states_from_episode')
    requests.get(base_url + '/run_cmd?cmd=use_reset_pool&to=demonstrations&name=random_states_from_episode')
    requests.get(base_url + '/run_cmd?cmd=set_demonstrations_reset_mode&mode=from_state_cache')


# Helper functions

def get_n_prefs(base_url):
    return int(requests.get(base_url + '/get_status').json()['No. prefs'])


def get_n_demos(base_url):
    return int(requests.get(base_url + '/get_status').json()['No. demonstrated episodes'])


def start_tmux_sess_with_cmd(sess_name, cmd):
    cmd += '; echo; read -p "Press enter to exit..."'
    cmd = ['tmux', 'new-sess', '-d', '-s', sess_name, '-n', f'main', cmd]
    subprocess.run(cmd)


def run_in_tmux_sess(sess_name, cmd, window_name):
    #window_name += '_' + str(uuid.uuid4())[:4]
    cmd += '; echo; read -p "Press enter to exit..."'
    tmux_cmd = ['tmux', 'new-window', '-ad', '-t', f'main', '-n', window_name, cmd] #-n makes new window
    subprocess.run(tmux_cmd)
    return window_name

def get_open_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

import socket
def forward_port(port, tmux_sess, run_seed):
    cmd = f"~/.ngrok http {port}" #requires ngrok to be installed at ~/.ngrok
    run_in_tmux_sess(tmux_sess, cmd, f"ngrok_{run_seed}")

    url = "http://localhost:4040/api/tunnels"
    connected = False
    while not connected:
        try:
            res = requests.get(url)
            res_unicode = res.content.decode("utf-8")
            res_json = json.loads(res_unicode)
            return res_json["tunnels"][0]["public_url"]
        except Exception as e:
            pass

if __name__ == '__main__':
    main()
