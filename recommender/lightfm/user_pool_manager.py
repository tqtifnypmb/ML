from user_generator import UserGenerator

class UserPoolManager:

    def __init__(self):
        self.ug = UserGenerator(10000)

    def pop_waiting_users(self):
        waiting_users = self.ug.retrieve_users()
        
        centroid_users = self._pick_centroid_users(waiting_users)
        return waiting_users, centroid_users

    def push_waiting_users(self, user_ids):
        self.ug.user_exit(user_ids)

    def _pick_centroid_users(self, users):
        pass

    def fake(self):
        self.ug.run()