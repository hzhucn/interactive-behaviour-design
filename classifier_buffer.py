#!/usr/bin/env python3

import argparse
import glob
import gzip
import lzma
import os
import pickle
import tempfile
import time
from os import path as osp
from threading import Thread
from typing import Dict, List

import magic
import numpy as np

TEST_FRACTION = 0.2


class LabelledObs:
    def __init__(self, time_step, obs, labels, validation):
        self.time_step = time_step
        self.labels = labels
        self.validation = validation
        self.compressed_obs = lzma.compress(pickle.dumps(obs))

    @property
    def obs(self):
        return pickle.loads(lzma.decompress(self.compressed_obs))


class LabelledObsList:
    def __init__(self, obses, vid_path):
        self.labelled_obses = []
        for t, obs in enumerate(obses):
            self.labelled_obses.append(LabelledObs(t, obs, {}, None))
        self.vid_path = vid_path

    def __getitem__(self, item):
        return self.labelled_obses[item]

    def __len__(self):
        return len(self.labelled_obses)


class ClassifierDataBuffer:
    episodes: Dict[int, LabelledObsList]

    def __init__(self, video_dir, save_dir):
        self.collector_thread = None
        self.save_thread = None
        self.episodes = {}
        self.num_episodes_from_exp_dir = 0
        self.seg_n = 0
        self.video_dir = video_dir
        self.save_dir = save_dir
        if save_dir:
            self.start_save_thread()

    def start_save_thread(self):
        def f():
            while True:
                time.sleep(30.0)
                self.save()

        self.save_thread = Thread(target=f)
        self.save_thread.start()

    def save(self):
        print("Saving experience buffer...")
        save_path = osp.join(self.save_dir, 'experience.pkl')
        fd, temp_file = tempfile.mkstemp(dir=self.save_dir)
        with open(temp_file, 'wb') as f:
            pickle.dump(self.episodes, f)
        os.close(fd)
        os.rename(temp_file, save_path)
        print("Experience buffer saved to '{}'".format(save_path))

    def load_from_dir(self, load_path):
        # if self.save_dir is not None:
        #     for vid_filename in glob.glob(osp.join(load_path, '*.mp4')):
        #         shutil.copy(vid_filename, self.save_dir)
        filename = osp.join(load_path, 'experience.pkl')
        if 'gzip' in magic.from_file(filename):
            open_f = gzip.open
        else:
            open_f = open
        with open_f(filename, 'rb') as f:
            self.episodes = pickle.load(f)
        # self.munge_experience()

    def rename_labels(self):
        for ep in self.episodes.values():
            for frame in ep.labelled_obses:
                if 'pos-up' in frame.labels:
                    del frame.labels['pos-up']
                if 'pos-up2' in frame.labels:
                    del frame.labels['pos-up2']
                if 'pos-top3' in frame.labels:
                    frame.labels['pos-top'] = frame.labels.pop('pos-top3')


    def start_saving_obs_from_queue(self, episode_queue):
        def f():
            while True:
                episode_n, obses = episode_queue.get()
                video_id = "video{:06}".format(episode_n)
                try:
                    vid_path = glob.glob(osp.join(self.video_dir, '*' + video_id + '.mp4'))[0]
                except IndexError:
                    print("Warning: video for episode {} not found".format(episode_n))
                    continue
                episode = LabelledObsList(obses, vid_path)
                suffix = ""
                if self.num_episodes_from_exp_dir > 0:
                    suffix = "*"
                self.episodes[str(episode_n + self.num_episodes_from_exp_dir) + suffix] = episode

        self.collector_thread = Thread(target=f)
        self.collector_thread.start()

    def reset_labels(self):
        for ep in self.episodes.values():
            for obs in ep.labelled_obses:
                obs.labels = {}
                obs.validation = None

    def tag(self, ep_name, frame_n, label_name, label):
        self.episodes[ep_name].labelled_obses[frame_n].labels[label_name] = label
        if np.random.rand() < TEST_FRACTION:
            validation = True
        else:
            validation = False
        self.episodes[ep_name].labelled_obses[frame_n].validation = validation

    def label_counts(self, label_name, ep_name=None):
        if ep_name:
            obses = self.episodes[ep_name]
        else:
            obses = self.get_all_obses()
        labels = [obs.labels[label_name]
                  for obs in obses
                  if label_name in obs.labels]
        labels_to_counts = dict(zip(*np.unique(labels, return_counts=True)))
        return labels_to_counts

    def get_all_obses(self):
        obses: List[LabelledObs] = []
        for ep in self.episodes.values():
            obses.extend(ep.labelled_obses)
        return obses

    def get_label_names(self):
        names = set()
        # make a copy with list() to avoid 'dictionary changed size during
        # iteration'
        for ep in list(self.episodes.values()):
            for obs in ep.labelled_obses:
                for name in obs.labels.keys():
                    names.add(name)
        return names

    def get_obses_with_labels(self, label_name, validation, shuffle=True):
        labelled_obses = self.get_all_obses()
        idxs = [n
                for n, lobs in enumerate(labelled_obses)
                if label_name in lobs.labels
                and lobs.validation == validation]
        if shuffle:
            np.random.shuffle(idxs)
        obses = []
        labels = []
        for idx in idxs:
            lobs = labelled_obses[idx]
            obses.append(lobs.obs)
            labels.append(lobs.labels[label_name])
        return obses, labels

    def newest_episode(self):
        return sorted(self.episodes.keys())[-1]

    def show(self):
        import matplotlib.pyplot as plt
        plt.ion()
        fig = plt.figure()
        plt.show()
        for ep_name, episode in self.episodes.items():
            for obs_n in range(len(episode)):
                obs = self.episodes[ep_name].labelled_obses[obs_n]
                if len(obs.labels) == 0:
                    continue
                print("Episode {}, frame {}: "
                      "shape {}, validation {}, labels {}".format(
                    ep_name, obs_n,
                    obs.obs.shape, obs.validation, obs.labels))
                im = obs.obs
                if im.shape in [(210, 160, 4), (84, 84, 4)]:
                    im = np.moveaxis(im, 2, 0)
                    im = np.hstack(im)
                    im = np.expand_dims(im, axis=2)
                    im = np.repeat(im, 3, axis=2)
                else:
                    print("Unsure how to deal with shape '{}'".format(im.shape))
                plt.imshow(im)
                input()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experience_dir')
    args = parser.parse_args()
    buf = ClassifierDataBuffer(None, None)
    buf.load_from_dir(args.experience_dir)
    buf.show()


if __name__ == '__main__':
    main()
