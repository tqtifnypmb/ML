from matrix_builder import UserItemMatrix
from repr_learner import RepresentationLearner
from user_matcher import UserMatcher
from user_generator import UserGenerator
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

        time.sleep(10)

class Recommender:

    def __init__(self, n_components=30):
        self.n_components = n_components
        self.repr_learner = RepresentationLearner.load('./model')

    def start(self):
        global PlayingUserPool
        
        while True:

            cur_users = ug.retrieve_users()

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