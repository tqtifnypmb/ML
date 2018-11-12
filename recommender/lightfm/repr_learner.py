from lightfm import LightFM
from matrix_builder import UserItemMatrix 

import pickle
import numpy as np
import pandas as pd
import scipy

class RepresentationLearner:

    def __init__(self, n_components=30):
        self.user_features = None
        self.item_features = None
        self.model = LightFM(n_components)

    def _merge_user_features(self, new_features):
        pass

    def _merge_item_features(self, new_features):
        pass

    def fit_partial(self, interactions, user_features=None, item_features=None):
        self._merge_user_features(user_features)
        self._merge_item_features(item_features)

        self.model.fit_partial(interactions,
                               user_features=user_features,
                               item_features=item_features)

    def user_representations(self):
        _, user_repr = self.model.get_user_representations()
        return user_repr

    def item_representations(self):
        _, item_repr = self.model.get_item_representations()
        return item_repr

    def save(self, path):
        with open(path, 'wb') as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as input:
            return pickle.load(input)

    def train(self, interaction_path, user_features_path=None, item_features_path=None):

        def read_fake_data(n_users, n_items, path):
            data = pd.read_csv(path)

            mat = scipy.sparse.lil_matrix((n_users, n_items), dtype=np.int32)

            for _, row in data.iterrows():
                userId, itemId, is_liked = row[0], row[1], row[2]
                mat[userId, itemId] = is_liked

            return mat

        n_users = 10000
        n_items = 10000
        interactions = read_fake_data(n_users, n_items, interaction_path)
        self.fit_partial(interactions)

# Unit test

if __name__ == "__main__":
    repr = RepresentationLearner()
    repr.train('interaction.csv')
    repr.save('./model')