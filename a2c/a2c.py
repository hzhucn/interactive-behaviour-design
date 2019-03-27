from typing import Dict

import easy_tf_log
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from easy_tf_log import tflog
from tensorflow import Operation

from a2c.common import tf_util
from a2c.common.runners import AbstractEnvRunner
from a2c.utils import Scheduler, make_path, find_trainable_variables
from a2c.utils import cat_entropy, mse
from a2c.utils import discount_with_dones
from policies.base_policy import PolicyTrainMode


class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs,
                 vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear',
                 ckpt_load_file=None):

        print("Making session")
        sess = tf_util.make_session()

        A = tf.placeholder(tf.int32, [None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])

        BC_A = tf.placeholder(tf.int32, [None])

        BC_COEF_T = tf.placeholder(tf.float32)
        BC_ENT_COEF_T = tf.placeholder(tf.float32)

        ENT_COEF_T = tf.placeholder(tf.float32)

        act_model = policy(sess, ob_space, ac_space, nbatch=nenvs, nsteps=None, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nbatch=None, nsteps=None, reuse=True)
        bc_model = policy(sess, ob_space, ac_space, nbatch=None, nsteps=None, reuse=True)
        self.nenvs = nenvs

        print("Creating losses")
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        train_entropy = tf.reduce_mean(cat_entropy(train_model.pi))

        bc_neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=bc_model.pi,
                                                                      labels=BC_A)
        bc_entropy = tf.reduce_mean(cat_entropy(bc_model.pi))
        bc_loss = tf.reduce_mean(bc_neglogpac)

        losses = dict()
        reward_loss = pg_loss - train_entropy * ENT_COEF_T + vf_loss * vf_coef
        bc_loss = bc_loss - bc_entropy * BC_ENT_COEF_T
        losses[PolicyTrainMode.R_ONLY] = reward_loss
        losses[PolicyTrainMode.BC_ONLY] = bc_loss
        losses[PolicyTrainMode.R_PLUS_BC] = reward_loss + BC_COEF_T * bc_loss

        optimizer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        params = find_trainable_variables("model")
        train_ops = dict()  # type: Dict[PolicyTrainMode, Operation]
        for loss in [PolicyTrainMode.R_ONLY, PolicyTrainMode.R_PLUS_BC, PolicyTrainMode.BC_ONLY]:
            grads = tf.gradients(losses[loss], params)
            if max_grad_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
            grads = list(zip(grads, params))
            train_ops[loss] = optimizer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        init_op = tf.global_variables_initializer()

        def train(obs, bc_obs, states, rewards, masks, actions, bc_actions, values,
                  loss_mode: PolicyTrainMode, bc_coef, ent_coef):
            if obs is not None:
                n_samples = len(obs)
            elif bc_obs is not None:
                n_samples = len(bc_obs)
            else:
                raise Exception("Neither obs nor bc_obs passed")

            for step in range(n_samples):
                cur_lr = lr.value()

            td_map = {LR: cur_lr,
                      BC_COEF_T: bc_coef,
                      ENT_COEF_T: ent_coef,
                      BC_ENT_COEF_T: self.bc_ent_coef}
            if loss_mode in [PolicyTrainMode.R_ONLY, PolicyTrainMode.R_PLUS_BC]:
                advs = rewards - values
                td_map.update({train_model.X: obs,
                               A: actions,
                               ADV: advs,
                               R: rewards,
                               LR: cur_lr})
            if loss_mode in [PolicyTrainMode.BC_ONLY, PolicyTrainMode.R_PLUS_BC]:
                td_map.update({
                    bc_model.X: bc_obs,
                    BC_A: bc_actions,
                })

            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            train_op = train_ops[loss_mode]

            if loss_mode == PolicyTrainMode.R_ONLY:
                rl_loss_val, policy_entropy, _ = sess.run([reward_loss, train_entropy, train_op], td_map)
                bc_loss_val = None
            elif loss_mode == PolicyTrainMode.BC_ONLY:
                bc_loss_val, _ = sess.run([bc_loss, train_op], td_map)
                policy_entropy = 0
                rl_loss_val = None
            elif loss_mode == PolicyTrainMode.R_PLUS_BC:
                rl_loss_val, bc_loss_val, policy_entropy, _ = sess.run([reward_loss, bc_loss, train_entropy, train_op], td_map)

            return rl_loss_val, bc_loss_val, policy_entropy

        def check_bc_loss(bc_obs, bc_actions):
            feed_dict = {bc_model.X: bc_obs, BC_A: bc_actions, BC_ENT_COEF_T: self.bc_ent_coef}
            return sess.run(bc_loss, feed_dict)

        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        def reset():
            sess.run(init_op)

        self.train = train
        self.check_bc_loss = check_bc_loss
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        self.sess = sess
        self.reset = reset
        self.params = params
        self.sess = sess
        self.bc_ent_coef = 0.1

        reset()
        if ckpt_load_file:
            saver = tf.train.Saver(tf.trainable_variables("model/core"))
            saver.restore(sess, ckpt_load_file)
            print("Loaded A2C core network checkpoint from '{}'".format(ckpt_load_file))


class Runner(AbstractEnvRunner):

    def __init__(self, env, model, nsteps=5, gamma=0.99):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        self.episode_n = [0] * env.num_envs
        easy_tf_log.set_dir(osp.join('/tmp', 'tfloga2c'))

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            # noinspection PyAssignmentToLoopOrWithParameter
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()

        r_raw = np.copy(mb_rewards)

        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards

        for n in range(self.nsteps):
            tflog('{}/val'.format(self.episode_n[0]), mb_values[0][n])
            tflog('{}/rewards'.format(self.episode_n[0]), r_raw[0][n])
            tflog('{}/returns'.format(self.episode_n[0]), mb_rewards[0][n])
            tflog('{}/adv'.format(self.episode_n[0]), mb_rewards[0][n] - mb_values[0][n])
            if mb_dones[0][n]:
                self.episode_n[0] += 1

        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values


