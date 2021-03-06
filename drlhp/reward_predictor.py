import logging
import os.path as osp
import time

import easy_tf_log
import numpy as np
import tensorflow as tf
from numpy.testing import assert_equal

from drlhp.pref_db import PrefDB
from drlhp.reward_predictor_core_network import net_cnn
from drlhp.drlhp_utils import LimitedRunningStat, RunningStat
from utils import batch_iter

MIN_L2_REG_COEF = 0.1


class RewardPredictor:

    def __init__(self, obs_shape, network, network_args, r_std, lr=1e-4, log_dir=None, seed=None):
        self.obs_shape = obs_shape
        graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=graph, config=config)

        with graph.as_default():
            if seed is not None:
                tf.set_random_seed(seed)
            self.l2_reg_coef = MIN_L2_REG_COEF
            self.rps = [RewardPredictorNetwork(core_network=network,
                                               network_args=network_args,
                                               obs_shape=obs_shape,
                                               lr=lr)]
            self.init_op = tf.global_variables_initializer()
            # Why save_relative_paths=True?
            # So that the plain-text 'checkpoint' file written uses relative paths,
            # which seems to be needed in order to avoid confusing saver.restore()
            # when restoring from FloydHub runs.
            self.saver = tf.train.Saver(max_to_keep=2, save_relative_paths=True)
            self.summaries = self.add_summary_ops()

        self.train_writer = tf.summary.FileWriter(
            osp.join(log_dir, 'reward_predictor', 'train'), flush_secs=5)
        self.test_writer = tf.summary.FileWriter(
            osp.join(log_dir, 'reward_predictor', 'test'), flush_secs=5)

        self.n_steps = 0
        self.r_norm_limited = LimitedRunningStat()
        self.r_norm = RunningStat(shape=[])
        self.r_std = r_std

        self.logger = easy_tf_log.Logger()
        self.logger.set_log_dir(osp.join(log_dir, 'reward_predictor', 'misc'))
        self.reward_call_n = 0

        self.log_interval = 20

        self.init_network()


    def add_summary_ops(self):
        summary_ops = []

        for pred_n, rp in enumerate(self.rps):
            name = 'reward_predictor/accuracy_{}'.format(pred_n)
            op = tf.summary.scalar(name, rp.accuracy)
            summary_ops.append(op)
            name = 'reward_predictor/loss_{}'.format(pred_n)
            op = tf.summary.scalar(name, rp.loss)
            summary_ops.append(op)
            l2_reg_losses = [rp.l2_reg_loss for rp in self.rps]
            mean_reg_loss = tf.reduce_mean(l2_reg_losses)
            op = tf.summary.scalar('reward_predictor/l2_loss_mean', mean_reg_loss)
            summary_ops.append(op)

        summaries = tf.summary.merge(summary_ops)

        return summaries

    def init_network(self, load_ckpt_dir=None):
        if load_ckpt_dir:
            ckpt_file = tf.train.latest_checkpoint(load_ckpt_dir)
            if ckpt_file is None:
                msg = "No reward predictor checkpoint found in '{}'".format(
                    load_ckpt_dir)
                raise FileNotFoundError(msg)
            self.saver.restore(self.sess, ckpt_file)
            print("Loaded reward predictor checkpoint from '{}'".format(ckpt_file))
        else:
            self.sess.run(self.init_op)

    def save(self, path):
        ckpt_name = self.saver.save(self.sess, path)
        print("Saved reward predictor checkpoint to '{}'".format(ckpt_name))

    def load(self, path):
        self.saver.restore(self.sess, path)
        print("Restored reward predictor from checkpoint '{}'".format(path))

    def raw_rewards(self, obs):
        """
        Return (unnormalized) reward for each frame of a single segment
        from each member of the ensemble.
        """
        assert_equal(obs.shape[1:], self.obs_shape)
        n_steps = obs.shape[0]
        feed_dict = self.get_feed_dict()
        for rp in self.rps:
            feed_dict[rp.training] = False
            feed_dict[rp.s1] = [obs]
        # This will return nested lists of sizes n_preds x 1 x nsteps
        # (x 1 because of the batch size of 1)
        rs = self.sess.run([rp.r1 for rp in self.rps], feed_dict)
        rs = np.array(rs)
        # Get rid of the extra x 1 dimension
        rs = rs[:, 0, :]
        n_preds = 1
        assert_equal(rs.shape, (n_preds, n_steps))
        return rs

    def reward(self, obs):
        """
        Return (normalized) reward for each frame of a single segment.

        (Normalization involves normalizing the rewards from each member of the
        ensemble separately, then averaging the resulting rewards across all
        ensemble members.)
        """
        assert_equal(obs.shape[1:], self.obs_shape)
        n_steps = obs.shape[0]

        # Get unnormalized rewards

        ensemble_rs = self.raw_rewards(obs)
        logging.debug("Unnormalized rewards:\n%s", ensemble_rs)

        # Normalize rewards

        # Note that we implement this here instead of in the network itself
        # because:
        # * It's simpler not to do it in TensorFlow
        # * Preference prediction doesn't need normalized rewards. Only
        #   rewards sent to the the RL algorithm need to be normalized.
        #   So we can save on computation.

        # Page 4:
        # "We normalized the rewards produced by r^ to have zero mean and
        #  constant standard deviation."
        # Page 15: (Atari)
        # "Since the reward predictor is ultimately used to compare two sums
        #  over timesteps, its scale is arbitrary, and we normalize it to have
        #  a standard deviation of 0.05"
        # Page 5:
        # "The estimate r^ is defined by independently normalizing each of
        #  these predictors..."

        # We want to keep track of running mean/stddev for each member of the
        # ensemble separately, so we have to be a little careful here.
        n_preds = 1
        assert_equal(ensemble_rs.shape, (n_preds, n_steps))
        ensemble_rs = ensemble_rs.transpose()
        assert_equal(ensemble_rs.shape, (n_steps, n_preds))
        for ensemble_rs_step in ensemble_rs:
            self.r_norm_limited.push(ensemble_rs_step[0])
            self.r_norm.push(ensemble_rs_step[0])
        ensemble_rs -= self.r_norm.mean
        ensemble_rs /= (self.r_norm.std + 1e-12)
        ensemble_rs *= self.r_std
        ensemble_rs = ensemble_rs.transpose()
        assert_equal(ensemble_rs.shape, (n_preds, n_steps))

        self.reward_call_n += 1
        if self.reward_call_n % 1000 == 0:
            self.logger.logkv('reward_predictor/r_norm_mean_recent', self.r_norm_limited.mean)
            self.logger.logkv('reward_predictor/r_norm_std_recent', self.r_norm_limited.std)
            self.logger.logkv('reward_predictor/r_norm_mean', self.r_norm.mean)
            self.logger.logkv('reward_predictor/r_norm_std', self.r_norm.std)

        # "...and then averaging the results."
        rs = np.mean(ensemble_rs, axis=0)
        assert_equal(rs.shape, (n_steps,))
        logging.debug("After ensemble averaging:\n%s", rs)

        return rs

    def train(self, prefs_train: PrefDB, prefs_val: PrefDB, val_interval, verbose=True):
        """
        Train all ensemble members for one epoch.
        """

        if verbose:
            print("Training/testing with %d/%d preferences" % (len(prefs_train), len(prefs_val)))

        start_steps = self.n_steps
        start_time = time.time()

        train_losses = []
        val_losses = []
        for _, batch in enumerate(batch_iter(prefs_train.prefs, batch_size=32, shuffle=True)):
            train_losses.append(self.train_step(batch, prefs_train))
            self.n_steps += 1

            if self.n_steps and self.n_steps % val_interval == 0 and len(prefs_val) != 0:
                val_losses.append(self.val_step(prefs_val))

        if val_losses:
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            ratio = val_loss / train_loss
            self.logger.logkv('reward_predictor/test_train_loss_ratio', ratio)
            if ratio > 1.3:
                self.l2_reg_coef *= 1.5
            elif ratio < 1.3:
                self.l2_reg_coef = max(self.l2_reg_coef / 1.5, MIN_L2_REG_COEF)
            self.logger.logkv('reward_predictor/reg_coef', self.l2_reg_coef)

        end_time = time.time()
        end_steps = self.n_steps
        rate = (end_steps - start_steps) / (end_time - start_time)
        self.logger.logkv('reward_predictor/training_steps_per_second', rate)
        if verbose:
            print("Done training DRLHP!")

    def train_step(self, batch, prefs_train):
        s1s = [prefs_train.segments[k1] for k1, k2, pref, in batch]
        s2s = [prefs_train.segments[k2] for k1, k2, pref, in batch]
        prefs = [pref for k1, k2, pref, in batch]
        feed_dict = self.get_feed_dict()
        for rp in self.rps:
            feed_dict[rp.s1] = s1s
            feed_dict[rp.s2] = s2s
            feed_dict[rp.pref] = prefs
            feed_dict[rp.training] = True
        # Why do we only check the loss from the first reward predictor?
        # As a quick hack to get adaptive L2 regularization working quickly,
        # assuming we're only using one reward predictor.
        ops = [self.rps[0].loss, self.summaries, [rp.train for rp in self.rps]]
        loss, summaries, _ = self.sess.run(ops, feed_dict)
        if self.n_steps % self.log_interval == 0:
            self.train_writer.add_summary(summaries, self.n_steps)
        return loss

    def val_step(self, prefs_val):
        val_batch_size = 32
        if len(prefs_val.prefs) <= val_batch_size:
            batch = prefs_val.prefs
        else:
            idxs = np.random.choice(len(prefs_val.prefs), val_batch_size, replace=False)
            batch = [prefs_val.prefs[i] for i in idxs]
        s1s = [prefs_val.segments[k1] for k1, k2, pref, in batch]
        s2s = [prefs_val.segments[k2] for k1, k2, pref, in batch]
        prefs = [pref for k1, k2, pref, in batch]
        feed_dict = self.get_feed_dict()
        for rp in self.rps:
            feed_dict[rp.s1] = s1s
            feed_dict[rp.s2] = s2s
            feed_dict[rp.pref] = prefs
            feed_dict[rp.training] = False
        loss, summaries = self.sess.run([self.rps[0].loss, self.summaries], feed_dict)
        if self.n_steps % self.log_interval == 0:
            self.test_writer.add_summary(summaries, self.n_steps)
        return loss

    def reset_normalisation(self):
        self.r_norm_limited = LimitedRunningStat()
        self.r_norm = RunningStat(shape=1)

    def get_feed_dict(self):
        feed_dict = {}
        for rp in self.rps:
            feed_dict[rp.l2_reg_coef] = self.l2_reg_coef
        return feed_dict


class RewardPredictorNetwork:
    """
    Predict the reward that a human would assign to each frame of
    the input trajectory, trained using the human's preferences between
    pairs of trajectories.

    Network inputs:
    - s1/s2     Trajectory pairs
    - pref      Preferences between each pair of trajectories
    Network outputs:
    - r1/r2     Reward predicted for each frame
    - rs1/rs2   Reward summed over all frames for each trajectory
    - pred      Predicted preference
    """

    def __init__(self, core_network, network_args, obs_shape, lr):
        training = tf.placeholder(tf.bool)
        obs_shape = tuple(obs_shape)
        # Each element of the batch is one trajectory segment.
        # (Dimensions are n segments x n frames per segment x ...)
        s1 = tf.placeholder(tf.float32, shape=(None, None) + obs_shape)
        s2 = tf.placeholder(tf.float32, shape=(None, None) + obs_shape)
        # For each trajectory segment, there is one human judgement.
        pref = tf.placeholder(tf.float32, shape=(None, 2))

        # Concatenate trajectory segments so that the first dimension is just
        # frames
        # (necessary because of conv layer's requirements on input shape)
        s1_unrolled = tf.reshape(s1, (-1,) + obs_shape)
        s2_unrolled = tf.reshape(s2, (-1,) + obs_shape)

        l2_reg_coef = tf.placeholder(tf.float32)
        l2_reg = tf.contrib.layers.l2_regularizer(scale=l2_reg_coef)
        # Predict rewards for each frame in the unrolled batch
        _r1 = core_network(s=s1_unrolled, reuse=False, training=training, regularizer=l2_reg,
                           **network_args)
        _r2 = core_network(s=s2_unrolled, reuse=True, training=training, regularizer=l2_reg,
                           **network_args)

        # Shape should be 'unrolled batch size'
        # where 'unrolled batch size' is 'batch size' x 'n frames per segment'
        c1 = tf.assert_rank(_r1, 1)
        c2 = tf.assert_rank(_r2, 1)
        with tf.control_dependencies([c1, c2]):
            # Re-roll to 'batch size' x 'n frames per segment'
            __r1 = tf.reshape(_r1, tf.shape(s1)[0:2])
            __r2 = tf.reshape(_r2, tf.shape(s2)[0:2])
        # Shape should be 'batch size' x 'n frames per segment'
        c1 = tf.assert_rank(__r1, 2)
        c2 = tf.assert_rank(__r2, 2)
        with tf.control_dependencies([c1, c2]):
            r1 = __r1
            r2 = __r2

        # Sum rewards over all frames in each segment
        _rs1 = tf.reduce_sum(r1, axis=1)
        _rs2 = tf.reduce_sum(r2, axis=1)
        # Shape should be 'batch size'
        c1 = tf.assert_rank(_rs1, 1)
        c2 = tf.assert_rank(_rs2, 1)
        with tf.control_dependencies([c1, c2]):
            rs1 = _rs1
            rs2 = _rs2

        # Predict preferences for each segment
        _rs = tf.stack([rs1, rs2], axis=1)
        # Shape should be 'batch size' x 2
        c1 = tf.assert_rank(_rs, 2)
        with tf.control_dependencies([c1]):
            rs = _rs
        _pred = tf.nn.softmax(rs)
        # Shape should be 'batch_size' x 2
        c1 = tf.assert_rank(_pred, 2)
        with tf.control_dependencies([c1]):
            pred = _pred

        preds_correct = tf.equal(tf.argmax(pref, 1), tf.argmax(pred, 1))
        accuracy = tf.reduce_mean(tf.cast(preds_correct, tf.float32))

        _loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=pref,
                                                           logits=rs)
        # Shape should be 'batch size'
        c1 = tf.assert_rank(_loss, 1)
        with tf.control_dependencies([c1]):
            loss = tf.reduce_sum(_loss)

        l2_reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # l2_reg_losses is a list of L2 norms - one for each weight layer
        # (where each L2 norm is just a scalar - so this is a list of scalars)
        # Why do we use add_n rather than reduce_sum?
        # reduce_sum is for when you have e.g. a matrix and you want to sum over one row.
        # If you want to sum over elements of a list, you use add_n.
        l2_reg_loss = tf.add_n(l2_reg_losses)
        loss += l2_reg_loss

        if core_network == net_cnn:
            batchnorm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            optimizer_dependencies = batchnorm_update_ops
        else:
            optimizer_dependencies = []

        with tf.control_dependencies(optimizer_dependencies):
            train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        # Inputs
        self.training = training
        self.s1 = s1
        self.s2 = s2
        self.pref = pref
        self.l2_reg_coef = l2_reg_coef

        # Outputs
        self.r1 = r1
        self.r2 = r2
        self.rs1 = rs1
        self.rs2 = rs2
        self.pred = pred

        self.accuracy = accuracy
        self.loss = loss
        self.train = train
        self.l2_reg_loss = l2_reg_loss
