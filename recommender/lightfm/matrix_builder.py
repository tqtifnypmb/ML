import pandas as pd
import numpy as np
import scipy.sparse

from sklearn.preprocessing import LabelEncoder

import datetime

class UserItemMatrix:

    def __init__(self, user_info_path, item_info_path):
        self.dataset = Dataset()

        self._build_user_info_dict(user_info_path)
        self._build_item_info_dict(item_info_path)
        

    def _build_user_info_dict(self, file_path):
        data = pd.read_table(file_path, delimiter='|', header=None)
        
        gender_encoder = LabelEncoder()
        gender_encoder.fit(data.iloc[:, 2])

        occupation_encoder = LabelEncoder()
        occupation_encoder.fit(data.iloc[:, 3])
        
        for _, row in data.iterrows():
            user_id, age, gender, occupation, zipcode = row[0], row[1], row[2], row[3], row[4]
            gender = gender_encoder.transform([gender])[0]
            occupation = occupation_encoder.transform([occupation])[0]

            self.dataset.fit_partial(users=[user_id],
                                     user_features=[(age, gender, occupation)])

    def _build_item_info_dict(self, file_path):
        data = pd.read_table(file_path, delimiter='|', header=None)

        for _, row in data.iterrows():
            item_id, release_str = row[0], row[2]

            if type(release_str) is not str:
                self.dataset.fit_partial(items=[item_id],
                                         item_features=[0])
            else:
                release_date = datetime.datetime.strptime(release_str, '%d-%b-%Y')
                release_date_int = int(release_date.strftime('%s'))

                self.dataset.fit_partial(items=[item_id],
                                         item_features=[release_date_int])
    
    def build_user_info_matrix(self, file_path):
        data = pd.read_table(file_path, delimiter='|', header=None)

        
    
    def load(self, file_path):
        data = pd.read_table(file_path, delimiter='\t', header=None)

        # update user rating
        for _, row in data.iterrows():
            userId, itemId, rating = row[0], row[1], row[2]

            self.dataset.build_intera
            is_like = rating > 2

            is_already_like = self.user_rating[userId][itemId - 1] == 1
            if not is_already_like:
                self.user_rating[userId][itemId - 1] = is_like

        users = []
        items = []
        user_features = []
        item_features = []

        for _, row in data.iterrows():
            userId, itemId = row[0], row[1]
            
            uf = self.user_info[userId]
            uf = np.append(uf, self.user_rating[userId])
            itemf = self.item_info[itemId]

            users.append(userId)
            items.append(itemId)
            user_features.append(uf)
            item_features.append(itemf)

        self.dataset.fit(users, items)


# Unit test

if __name__ == '__main__':
    uimatrix = UserItemMatrix('../data/ml-100k/u.user', '../data/ml-100k/u.item')
    uimatrix.load('../data/ml-100k/u1.base')