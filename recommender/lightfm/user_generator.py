import random
import time
import thread
from threading import Lock

UserPool = []
UserPoolLock = Lock()

class UserGenerator:
    def __init__(self, n_users):
        self.users = [x for x in xrange(n_users)]
        self.usersLock = Lock()

    def run(self):
        global UserPool
        global UserPoolLock

        while True:
            UserPoolLock.acquire()
            self.usersLock.acquire()

            if len(self.users) > 0:
                n_arrive_users = random.randint(0, min(1000, len(self.users)))
                random.shuffle(self.users)
                UserPool += self.users[:n_arrive_users]
                self.users = self.users[n_arrive_users:]

                print('generated {}'.format(len(UserPool)))

            self.usersLock.release()
            UserPoolLock.release()

            interval = random.randint(0, 3)
            time.sleep(interval)

    def user_exit(self, user):
        self.usersLock.acquire()

        self.users += user

        print("remain users: {}".format(len(self.users)))
        
        self.usersLock.release()
    
# Unit test

if __name__ == "__main__":
    
    ug = UserGenerator(100000)

    thread.start_new_thread(ug.run, ())
    
    while True:
        print(UserPool)
        time.sleep(3)