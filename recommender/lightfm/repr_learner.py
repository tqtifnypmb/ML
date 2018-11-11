from lightfm import LightFM
from matrix_builder import UserItemMatrix 

import pickle

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