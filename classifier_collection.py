import os.path as osp
from collections import namedtuple

import numpy as np
import tensorflow as tf

from classifier_buffer import ClassifierDataBuffer
from a2c.utils import find_trainable_variables
from utils import batch_iter

class Classifier:
    def __init__(self, pred, probs, pred_prob, acc, loss, train):
        self.pred = pred
        self.probs = probs
        self.pred_prob = pred_prob
        self.acc = acc
        self.loss = loss
        self.train = train

class ClassifierCollection:

    experience_buffer: ClassifierDataBuffer

    def __init__(self, experience_buffer, log_dir, network_fn, obs_shape):

        self.experience_buffer = experience_buffer

        self.graph = tf.Graph()
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            images = tf.placeholder(tf.float32, shape=[None] + list(obs_shape))
            self.images = images / 255.0
            self.labels = tf.placeholder(tf.int64, shape=(None,))
            self.training = tf.placeholder(tf.bool)

        self.network_fn = network_fn
        self.classifiers = {}
        self.init_ops = {}

        self.n_steps = 0

    def get_labels(self):
        return self.classifiers.keys()

    def reset(self, label_name):
        self.sess.run(self.init_ops[label_name])

    def add_classifier(self, label_name):
        if label_name in self.classifiers:
            raise Exception("Classifier '{}' already exists".format(label_name))
        if 'pos' not in label_name and 'neg' not in label_name:
            raise Exception("Unknown polarity for '{}'".format(label_name))

        scope = "classifier-{}".format(label_name)
        with self.graph.as_default():
            with tf.variable_scope(scope):
                features = self.network_fn(self.images, 'classifier')
                features = tf.layers.dropout(features, rate=0.5, training=self.training)
                logits = tf.layers.dense(inputs=features, units=2)
                prediction = tf.argmax(logits, axis=1)
                correct_prediction = tf.equal(prediction, self.labels)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                probs = tf.nn.softmax(logits, axis=1)
                pred_prob = tf.reduce_max(probs, axis=1)
                loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=logits)
                loss = tf.reduce_mean(loss)
                optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
                train_op = optimizer.minimize(loss)

            init_op = tf.variables_initializer(tf.global_variables(scope))
            self.sess.run(init_op)
            self.saver = tf.train.Saver(max_to_keep=2)

            params = tf.trainable_variables()
            self.saver = tf.train.Saver(params, max_to_keep=2)

        self.init_ops[label_name] = init_op
        self.classifiers[label_name] = Classifier(pred=prediction,
                                                  probs=probs,
                                                  pred_prob=pred_prob,
                                                  acc=accuracy,
                                                  loss=loss,
                                                  train=train_op)

    @staticmethod
    def balance_positive_negative(labels):
        """
        Take in a possibly unbalanced list of labels, and sample labels
        (randomly without replacement) from that list to create a balanced
        list. Positive and negative examples are interleaved in the resulting
        list, suitable for splitting up into batches which are also balanced.

        We do that by dropping surplus labels from whichever side has more.
        We don't actually create a new list, but rather return the indices of
        the supplied list we would use to create the new list, so that the
        reordering can also be applied to the images.
        """
        negative_idxs = np.argwhere(np.array(labels) == 0).flatten()
        positive_idxs = np.argwhere(np.array(labels) == 1).flatten()

        np.random.shuffle(negative_idxs)
        np.random.shuffle(positive_idxs)

        min_len = min(len(negative_idxs), len(positive_idxs))
        negative_idxs = negative_idxs[:min_len]
        positive_idxs = positive_idxs[:min_len]

        idxs = [i for tup in zip(positive_idxs, negative_idxs) for i in tup]
        return idxs

    def train(self, label_name):
        images, labels = self.experience_buffer.get_obses_with_labels(label_name, validation=False)
        idxs = self.balance_positive_negative(labels)
        images = [images[i] for i in idxs]
        labels = [labels[i] for i in idxs]
        train_data = list(zip(images, labels))

        print("Training with {} examples".format(len(labels)))

        cls = self.classifiers[label_name]
        batch_losses = []
        batch_accs = []

        for batch in batch_iter(train_data, batch_size=16):
            batch_images, batch_labels = zip(*batch)
            feed_dict = {
                self.training: True,
                self.images: batch_images,
                self.labels: batch_labels
            }
            _, loss, acc = self.sess.run([cls.train,
                                          cls.loss,
                                          cls.acc],
                                         feed_dict)
            batch_losses.append(loss)
            batch_accs.append(acc)

        loss = sum(batch_losses) / len(batch_losses)
        acc = sum(batch_accs) / len(batch_accs)
        print("Train loss/accuracy: {:.3f}/{:.2f}".format(loss, acc))
        self.n_steps += 1
        print("Trained for {} steps".format(self.n_steps))

    def test(self, label_name):
        frames, labels = self.experience_buffer.get_obses_with_labels(label_name, validation=True)

        print("Testing with {} examples".format(len(frames)))

        cls = self.classifiers[label_name]
        loss, acc = self.sess.run([cls.loss, cls.acc],
                                  feed_dict={self.training: False,
                                             self.images: frames,
                                             self.labels: labels})
        print("Test loss/accuracy: {:.3f}/{:.2f}".format(loss, acc))

    def save_checkpoint(self, path):
        if not self.classifiers:
            return
        saved_path = self.saver.save(self.sess, path)
        print("Saving classifiers:", list(self.classifiers.keys()))
        print("Saved classifier checkpoint to '{}'".format(saved_path))

    def load_checkpoint(self, path):
        self.saver.restore(self.sess, path)
        print("Restored classifier checkpoint from '{}'".format(path))

    def predict(self, label_name, images):
        cls = self.classifiers[label_name]
        preds = self.sess.run(cls.pred,
                              feed_dict={self.images: images,
                                         self.training: False})
        assert preds.shape == (len(images),)
        return preds

    def predict_positive_prob(self, label_name, images):
        cls = self.classifiers[label_name]
        probs = self.sess.run(cls.probs,
                              feed_dict={self.images: images,
                                         self.training: False})
        assert probs.shape == (len(images), 2)
        return probs[:, 1]

    def predict_positive_probs(self, images):
        ops = []
        label_names = self.experience_buffer.get_label_names()
        for label_name in label_names:
            cls = self.classifiers[label_name]
            ops.append(cls.probs)
        probs = self.sess.run(ops,
                              feed_dict={self.images: images,
                                         self.training: False})
        probs = np.array(probs)
        assert probs.shape == (len(label_names), len(images), 2)
        probs = probs[:, :, 1]
        assert probs.shape == (len(label_names), len(images))

        return list(zip(label_names, probs))

    def predict_probs_mc(self, label_name, images):
        cls = self.classifiers[label_name]

        n_repeats = 10
        images_repeated = np.repeat(images, axis=0, repeats=n_repeats)
        probs = self.sess.run(cls.probs,
                              feed_dict={self.images: images_repeated,
                                         self.training: True})

        assert probs.shape == (len(images) * n_repeats, 2)
        probs = np.reshape(probs, [len(images), n_repeats, 2])

        return probs

    def predict_with_confidence(self, label_name, images):
        probs = self.predict_probs_mc(label_name, images)

        probs = np.mean(probs, axis=1)
        assert probs.shape == (len(images), 2)

        preds = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        preds_with_confidences = list(zip(preds, confidences))
        return preds_with_confidences

    def predict_with_uncertainty(self, label_name, images):
        cls = self.classifiers[label_name]
        preds = self.sess.run(cls.pred,
                              feed_dict={self.images: images,
                                         self.training: False})

        probs = self.predict_probs_mc(label_name, images)
        uncertainties = variation_ratios(probs)

        preds_with_uncertainties = list(zip(preds, uncertainties))
        return preds_with_uncertainties


def variation_ratios(probs):
    # Mean over repeats
    mean_probs = np.mean(probs, axis=1)
    # Select max prob
    pred_probs = np.max(mean_probs, axis=1)
    return 1 - pred_probs
