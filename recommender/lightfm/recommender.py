from matrix_builder import UserItemMatrix
from repr_learner import RepresentationLearner
from user_matcher import UserMatcher
from user_generator import UserPool, UserPoolLock, UserGenerator
from threading import Lock

import scipy
import pandas as pd
import numpy as np
import thread
import time
import sets

ug = UserGenerator(10000)

PlayingUserPool = []
PlayingUserLock = Lock()

def release_user():
    print('realease user start')

    global PlayingUserPool

    while True:
        PlayingUserLock.acquire()

        if len(PlayingUserPool) > 0:
            print('release: {}'.format(len(PlayingUserPool)))
            ug.user_exit(PlayingUserPool)
            PlayingUserPool = []

        PlayingUserLock.release()

        time.sleep(5)

def read_fake_data(n_users, n_items, path):
    data = pd.read_csv(path)

    mat = scipy.sparse.lil_matrix((n_users, n_items), dtype=np.int32)

    for _, row in data.iterrows():
        userId, itemId, is_liked = row[0], row[1], row[2]
        mat[userId, itemId] = is_liked

    return mat

class Recommender:

    def __init__(self, n_components=30):
        self.n_components = n_components
        self.repr_learner = RepresentationLearner(n_components)
        
    def fit_partial(self, n_users, n_items, interaction_path, user_features_path=None, item_features_path=None):
        # mb = UserItemMatrix()
    
        # user_info = mb.build_user_info_matrix(user_features_path)
        # item_info = mb.build_item_info_matrix(item_features_path)

        # n_users = user_info.shape[0]
        # n_items = item_info.shape[0]

        # interactions = mb.build_interactions_matrix(n_users, n_items, interaction_path)

        print('reading data')

        interactions = read_fake_data(n_users, n_items, interaction_path)

        print('training')

        self.repr_learner.fit_partial(interactions)

    def start(self):
        global UserPool

        n_users = 10000
        n_items = 10000

        print('start training')

        self.fit_partial(n_users, n_items, 'interaction.csv')

        print('finish training')

        global PlayingUserPool

        while True:
            UserPoolLock.acquire()

            print('current users: {}'.format(len(UserPool)))
            cur_users = []
            cur_users += UserPool
            UserPool = []

            UserPoolLock.release()

            if len(cur_users) == 0:
                print('no users')
                time.sleep(5)
            elif len(cur_users) == 6:
                print('match randomly: {}'.format(cur_users))
                time.sleep(5)

                PlayingUserLock.acquire()
                PlayingUserPool += cur_users
                PlayingUserLock.release()
            else:
                user_matcher = UserMatcher(10, self.n_components)

                user_repr = self.repr_learner.user_representations()
                cur_users_repr = user_repr[cur_users]
        
                user_matcher.add_embedding(cur_users_repr, user_ids=np.array(cur_users))
                user_matcher.finish()

                matched_users = set()

                for user in cur_users:
                    if user in matched_users:
                        continue

                    matched = user_matcher.pick(user, 12)
                    ret = []

                    for i in matched:
                        if i in matched_users:
                            continue

                        ret.append(i)

                        if len(ret) == 6:
                            break

                    if len(ret) == 6:
                        for i in ret:
                            matched_users.add(i)

                        print('match: {}'.format(ret))

                    if len(ret) > 0:
                        PlayingUserLock.acquire()
                        PlayingUserPool += ret
                        PlayingUserLock.release()


if __name__ == "__main__":
    
    thread.start_new_thread(ug.run, ())
    thread.start_new_thread(release_user, ())

    rec = Recommender()
    rec.start()