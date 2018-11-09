import pandas as pd
import numpy as np
import scipy.sparse

from sklearn.preprocessing import LabelEncoder

import datetime

class UserItemMatrix:
    
    def build_user_info_matrix(self, file_path):
        data = pd.read_table(file_path, delimiter='|', header=None)
        n_users = len(data)
        n_features = 3

        mat = scipy.sparse.lil_matrix((n_users, n_features), dtype=np.int32)

        gender_encoder = LabelEncoder()
        gender_encoder.fit(data.iloc[:, 2])

        occupation_encoder = LabelEncoder()
        occupation_encoder.fit(data.iloc[:, 3])
        for _, row in data.iterrows():
            user_id, age, gender, occupation, zipcode = row[0], row[1], row[2], row[3], row[4]
            gender = gender_encoder.transform([gender])[0]
            occupation = occupation_encoder.transform([occupation])[0]

            mat[user_id - 1, :] = [age, gender, occupation]

        return mat.tocsr()

    def build_item_info_matrix(self, file_path):
        data = pd.read_table(file_path, delimiter='|', header=None)
        n_items = len(data)
        n_features = 1

        mat = scipy.sparse.lil_matrix((n_items, n_features), dtype=np.int32)
        for _, row in data.iterrows():
            item_id, release_str = row[0], row[2]

            if type(release_str) is not str:
                mat[item_id - 1, :] = [0]
            else:
                release_date = datetime.datetime.strptime(release_str, '%d-%b-%Y')
                release_date_int = int(release_date.strftime('%s'))

                mat[item_id - 1, :] = [release_date_int]

        return mat.tocsr()

    def build_interactions_matrix(self, rows, cols, file_path):
        data = pd.read_table(file_path, delimiter='\t', header=None)

        mat = scipy.sparse.lil_matrix((rows, cols), dtype=np.int32)

        for _, row in data.iterrows():
            userId, itemId, rating = row[0], row[1], row[2]
            mat[userId - 1, itemId - 1] = rating

        return mat.tocoo()

# Unit test

if __name__ == '__main__':
    uimatrix = UserItemMatrix()
    user_info = uimatrix.build_user_info_matrix('../data/ml-100k/u.user')
    item_info = uimatrix.build_item_info_matrix('../data/ml-100k/u.item')

    n_users = user_info.shape[0]
    n_items = item_info.shape[0]
    interactions = uimatrix.build_interactions_matrix(n_users, n_items, '../data/ml-100k/u1.base')

    print(interactions)