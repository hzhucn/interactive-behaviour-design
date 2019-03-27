import numpy as np
import tensorflow as tf
from a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from a2c.common.distributions import make_pdtype
from a2c.common.input import observation_input


def nature_cnn(unscaled_images, name="", **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    print("scaled images")
    activ = tf.nn.relu
    print("activ")
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    print("h")
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    print("h2")
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    print("h3")
    h3 = conv_to_fc(h3)
    print("h3'")
    h4 = activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))
    print("h4")
    return h4


class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        X, processed_x = observation_input(ob_space, nbatch)
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(processed_x)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value


class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)

        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False,
                 **conv_kwargs):  # pylint: disable=W0613
        print("Making Cnnpolicy")
        self.pdtype = make_pdtype(ac_space)
        print("Make pdtyp")
        X, processed_x = observation_input(ob_space, batch_size=nbatch)
        print("Made X")
        with tf.variable_scope("model", reuse=reuse):
            print("var scope")
            with tf.variable_scope("core", reuse=reuse):
                print("var scope)")
                h = nature_cnn(processed_x, **conv_kwargs)
                print("Made nature cnn")
            vf = fc(h, 'v', 1)[:, 0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp, logits = sess.run([a0, vf, neglogp0, self.pd.logits], {X: ob})
            if 'softmax_temp' in _kwargs:
                temperature = float(_kwargs['softmax_temp'])
                a = [np.random.choice(len(probs), p=probs) for probs in softmax(logits, temperature)]
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value

def softmax(x, temp):
    """Compute softmax values for each sets of scores in x with temperature temp."""
    x_temp = np.dot(1 / (temp + 1e-5), x)
    x_temp -= np.max(x_temp)
    e_x = np.exp(x_temp)
    softmax = e_x / e_x.sum(axis=1)[:,None]
    return softmax

def test_softmax():
    def cmp_softmax(temp):
        #flexing my new tf knowledge
        X = tf.placeholder(tf.float32, [1, 784])
        W = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1))
        b = tf.Variable(tf.ones([10]) / 10)
        logits = tf.matmul(X, W) + b
        Y = tf.nn.softmax(tf.div(logits, temp))

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        expected_softmax, lgts = sess.run([Y, logits], feed_dict={X: np.random.rand(1, 784)})
        expected_softmax = np.round(expected_softmax.astype(np.float64), 4) #make it the same type

        actual_softmax = np.round(softmax(lgts, temp), 4)
        return np.array_equal(expected_softmax, actual_softmax)

    test_outcomes = []
    for temp in range(1, 10):
        test_outcomes += [cmp_softmax(temp)]
    return all(test_outcomes) #all true?

def mlp(obs, name):
    activ = tf.tanh
    h1 = activ(fc(obs, '{}_fc1'.format(name), nh=64, init_scale=np.sqrt(2)))
    h2 = activ(fc(h1, '{}_fc2'.format(name), nh=64, init_scale=np.sqrt(2)))
    return h2


class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps,
                 reuse=False):  # pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            X, processed_x = observation_input(ob_space, nbatch)
            processed_x = tf.layers.flatten(processed_x)
            vf_h = mlp(processed_x, 'vf')
            pi_h = mlp(processed_x, 'pi')
            vf = fc(vf_h, 'vf', 1)[:, 0]
            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
