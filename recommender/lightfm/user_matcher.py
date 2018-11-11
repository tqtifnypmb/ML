from annoy import AnnoyIndex

class UserMatcher:

    def __init__(self, n_trees, n_dimensions):
        self.n_trees = n_trees
        self.model = AnnoyIndex(n_dimensions)

    def add_embedding(self, embeddings, user_ids=None):
        if user_ids is not None:
            assert(user_ids.shape[0] == embeddings.shape[0])

            for i in xrange(embeddings.shape[0]):
                embedding = embeddings[i]
                user_id = user_ids[i]
                self.model.add_item(user_id, embedding)
        else:
            for i in xrange(embeddings.shape[0]):
                embedding = embeddings[i]
                self.model.add_item(i, embedding)

    def finish(self):
        self.model.build(self.n_trees)

    def pick(self, center_user_id, n_neighbours):
        return self.model.get_nns_by_item(center_user_id, n_neighbours)

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass