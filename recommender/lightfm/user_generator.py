import random
import time
from threading import Lock

class UserGenerator:
    def __init__(self, n_users):
        self.users = [x for x in xrange(n_users)]
        self.users_lock = Lock()

        self.user_pool = []
        self.user_pool_lock = Lock()

    def run(self):
        while True:
            self.user_pool_lock.acquire()
            self.users_lock.acquire()

            if len(self.users) > 0:
                n_arrive_users = random.randint(0, min(1000, len(self.users)))
                random.shuffle(self.users)
                self.user_pool += self.users[:n_arrive_users]
                self.users = self.users[n_arrive_users:]

                print('generated {}'.format(len(self.user_pool)))

            self.users_lock.release()
            self.user_pool_lock.release()

            interval = random.randint(0, 5)
            time.sleep(interval)

    def user_exit(self, user):
        self.users_lock.acquire()

        self.users += user

        print("remain users: {}".format(len(self.users)))
        
        self.users_lock.release()

    def retrieve_users(self):
        self.user_pool_lock.acquire()

        cur_users = self.user_pool
        self.user_pool = []

        self.user_pool_lock.release()

        return cur_users