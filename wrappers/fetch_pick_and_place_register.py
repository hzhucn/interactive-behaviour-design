"""
Imports are done strangely here so that we can quickly query all the environment names for train_all.py without having to import anything
"""
from wrappers.fetch_pick_and_place import FixedGoal, FixedBlockInitialPositions, PickOnly


def make_env(repeat_n, early_termination, binary_gripper, grip_close_bonus, grip_open_bonus,
             reward_mode, n_initial_block_positions, fixed_goal, slow_gripper, vanilla_rl, full_obs, pick_only):
    from gym.envs.robotics import FetchPickAndPlaceEnv
    from gym.wrappers import FlattenDictWrapper

    from wrappers.fetch_pick_and_place import FetchStatsWrapper, FetchPickAndPlaceRewardWrapper, \
        FetchPickAndPlaceObsWrapper
    from wrappers.util_wrappers import RepeatActions

    env = FetchPickAndPlaceEnv()
    env = FixedGoal(env, fixed_goal=fixed_goal)
    env = FixedBlockInitialPositions(env, n_initial_block_positions=n_initial_block_positions)
    env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
    env = FetchStatsWrapper(env)
    env = FetchPickAndPlaceRewardWrapper(env,
                                         early_termination=early_termination,
                                         binary_gripper=binary_gripper,
                                         grip_close_bonus=grip_close_bonus,
                                         grip_open_bonus=grip_open_bonus,
                                         reward_mode=reward_mode,
                                         slow_gripper=slow_gripper,
                                         vanilla_rl=vanilla_rl)
    if pick_only:
        env = PickOnly(env)
    if full_obs:
        mode = 'full'
        include_grip_obs = True
    else:
        mode = 'minimal'
        include_grip_obs = False
    env = FetchPickAndPlaceObsWrapper(env, mode=mode, include_grip_obs=include_grip_obs)
    env = RepeatActions(env, repeat_n)
    return env


def enumerate_envs(register=False):
    repeat_ns = [('Repeat1', 1), ('Repeat3', 3)]
    binary_grippers = [('BinaryGripper', True), ('ContGripper', False)]
    gripper_bonuses = [('GripperBonus', True), ('NoGripperBonus', False)]
    early_terminations = [('ET', True), ('NoET', False)]
    n_initial_block_positionss = [('1InitialBlockPos', 1),
                                  ('5InitialBlockPos', 5),
                                  ('InfInitialBlockPos', None)]
    fixed_goals = [('FixedGoal', True), ('RandomGoal', False)]
    slow_grippers = [('SlowGripper', True), ('FastGripper', False)]
    vanilla_rls = [('VanillaRL', True), ('NoVanillaRL', False)]
    full_obss = [('FullObs', True), ('PartialObs', False)]
    modes = [('Delta', 'delta2'), ('NonDelta', 'nondelta')]
    pick_onlys = [('PickOnly', True), ('PickAndPlace', False)]

    env_ids = []
    for s1, repeat in repeat_ns:
        for s2, binary_gripper in binary_grippers:
                for s4, n_initial_block_positions in n_initial_block_positionss:
                    for s5, fixed_goal in fixed_goals:
                        for s6, gripper_bonus in gripper_bonuses:
                            for s7, early_termination in early_terminations:
                                for s8, slow_gripper in slow_grippers:
                                    for s9, vanilla_rl in vanilla_rls:
                                        for s10, full_obs in full_obss:
                                            for s11, mode in modes:
                                                for s12, pick_only in pick_onlys:
                                                    env_id = f'FetchPickAndPlace-{s1}-{s2}-{s4}-{s5}-{s6}-{s7}-{s8}-{s9}-{s10}-{s11}-{s12}-v0'
                                                    env_ids.append(env_id)
                                                    if register:
                                                        from gym.envs import register as gym_register
                                                        entry_fn = lambda locals=dict(locals()): \
                                                            make_env(reward_mode=locals['mode'],
                                                                     early_termination=locals['early_termination'],
                                                                     grip_open_bonus=locals['gripper_bonus'],
                                                                     grip_close_bonus=locals['gripper_bonus'],
                                                                     binary_gripper=locals['binary_gripper'],
                                                                     repeat_n=locals['repeat'],
                                                                     n_initial_block_positions=locals['n_initial_block_positions'],
                                                                     fixed_goal=locals['fixed_goal'],
                                                                     slow_gripper=locals['slow_gripper'],
                                                                     vanilla_rl=locals['vanilla_rl'],
                                                                     full_obs=locals['full_obs'],
                                                                     pick_only=locals['pick_only'])
                                                        gym_register(env_id,
                                                                     entry_point=entry_fn,
                                                                     max_episode_steps=250)
    return env_ids


def register():
    enumerate_envs(register=True)
