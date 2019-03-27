import datetime
import os
import threading
import time
from threading import Thread

import global_constants
from classifier_collection import ClassifierCollection
from drlhp.reward_predictor import RewardPredictor
from policies.policy_collection import PolicyCollection


class Checkpointer:
    def __init__(self, ckpt_dir: str,
                 policy_collection: PolicyCollection,
                 drlhp_reward_predictor: RewardPredictor,
                 classifier_collection: ClassifierCollection):
        self.ckpt_dir = ckpt_dir
        self.policy_collection = policy_collection
        self.drlhp_reward_predictor = drlhp_reward_predictor
        self.classifier_collection = classifier_collection
        self.lock = threading.Lock()

        def checkpoint_loop():
            while True:
                time.sleep(global_constants.CKPT_EVERY_N_SECONDS)
                self.checkpoint()
        Thread(target=checkpoint_loop).start()

    def checkpoint(self):
        self.lock.acquire()

        now = str(datetime.datetime.now())

        for policy_name in dict(self.policy_collection.policies):  # make a copy in case changed while iterating
            policy_ckpt_path = os.path.join(self.ckpt_dir,
                                            'policy-{}-{}.ckpt'.format(policy_name, now))
            policy = self.policy_collection[policy_name]
            policy.save_checkpoint(policy_ckpt_path)

        if len(self.classifier_collection.classifiers) > 0: #exist classifiers to save
            classifier_ckpt_path = os.path.join(self.ckpt_dir, 'classifiers-{}.ckpt'.format(now))
            self.classifier_collection.save_checkpoint(classifier_ckpt_path)

            #also save the classifier names for restoration
            classifier_names_path = os.path.join(self.ckpt_dir, 'classifier_names.txt'.format(now))
            f = open(classifier_names_path, "w") #clear file (overwrite)
            for classifier_name in self.classifier_collection.classifiers:
                f = open(classifier_names_path, "a")  # append to file
                f.write(classifier_name + "\n")
            f.close()

        rp_ckpt_path = os.path.join(self.ckpt_dir, 'drlhp_reward_predictor-{}.ckpt'.format(now))
        self.drlhp_reward_predictor.save(rp_ckpt_path)

        classifier_ckpt_path = os.path.join(self.ckpt_dir, 'classifiers-{}.ckpt'.format(now))
        self.classifier_collection.save_checkpoint(classifier_ckpt_path)
        # also save the classifier names for restoration
        classifier_names_path = os.path.join(self.ckpt_dir, 'classifier_names.txt'.format(now))
        f = open(classifier_names_path, "w")  # clear file (overwrite)
        for classifier_name in self.classifier_collection.classifiers:
            f = open(classifier_names_path, "a")  # append to file
            f.write(classifier_name + "\n")
        f.close()

        self.lock.release()
