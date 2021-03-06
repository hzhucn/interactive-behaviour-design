#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', default='0')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

seeds = list(map(int, args.seeds.split(',')))
if args.test:
    test_args = '--n_initial_prefs 0 --n_initial_demos 0 --pretrain_reward_predictor_seconds 0 --min_label_interval_seconds 0'
else:
    test_args = ''

rl_envs = [
    ('seaquest', 'SeaquestDeepMind-v0'),
    ('fetch', 'FetchPickAndPlace-Repeat1-BinaryGripper-5InitialBlockPos-FixedGoal-NoGripperBonus-NoET-FastGripper-VanillaRL-FullObs-Delta-WithGripObs-v0'),
    ('lunarlander', 'LunarLanderStatefulStats-v0'),
]

prefs_envs = [
    ('seaquest', 'SeaquestDeepMindDense-v0'),
    ('fetchpp', 'FetchPickAndPlace-Repeat1-BinaryGripper-5InitialBlockPos-FixedGoal-GripperBonus-NoET-SlowGripper-NoVanillaRL-PartialObs-NonDelta-WithGripObs-v0'),
    ('lunarlander', 'LunarLanderStatefulStats-v0'),
]

# RL using environment reward
for seed in seeds:
    for env_shortname, env_id in rl_envs:
        run_name = f"{env_shortname}-{seed}-rl"
        print(f"python3 scripts/train/auto_train_rl.py {seed} {env_id} {run_name}")

for seed in seeds:
    for env_shortname, env_id in prefs_envs:
        run_name = f"{env_shortname}-{seed}"
        if args.test:
            run_name += '-test'

        if 'Fetch' in env_id:
            # The speed of the Fetch subpolicies is set assuming Repeat1
            assert 'Repeat1' in env_id
            rollout_length_seconds = 0.15
        else:
            rollout_length_seconds = 1.0

        extra_args = f"--rollout_length_seconds {rollout_length_seconds}"

        # DRLHP
        print("python3 scripts/train/auto_train_prefs.py "
              f"{env_id} reward_only drlhp {run_name}-drlhp --seed {seed} --disable_redo --extra_args ' {extra_args}' {test_args}")
        # SDRLHP
        print("python3 scripts/train/auto_train_prefs.py "
              f"{env_id} reward_only demonstrations {run_name}-sdrlhp --seed {seed} --disable_redo --extra_args ' {extra_args}' {test_args}")

        if 'lunarlander' in env_shortname or 'fetch' in env_shortname:
            redo = '--disable_redo'
        else:
            redo = ''
        # Behavioral cloning
        print("python3 scripts/train/auto_train_prefs.py "
              f"{env_id} bc_only demonstrations {run_name}-bc --seed {seed} {redo} --extra_args ' {extra_args}' {test_args}")
        # SDRLHP + behavioral cloning
        print("python3 scripts/train/auto_train_prefs.py "
              f"{env_id} reward_plus_bc demonstrations {run_name}-sdrlhp-bc --seed {seed} {redo} --extra_args ' {extra_args}' {test_args}")
