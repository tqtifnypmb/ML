from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k
from matrix_builder import UserItemMatrix 
from annoy import AnnoyIndex

import pandas as pd
import numpy as np
import time
    
if __name__ == '__main__':
    
    mb = UserItemMatrix()
    
    user_info = mb.build_user_info_matrix('../data/ml-100k/u.user')
    item_info = mb.build_item_info_matrix('../data/ml-100k/u.item')

    n_users = user_info.shape[0]
    n_items = item_info.shape[0]

    train = mb.build_interactions_matrix(n_users, n_items, '../data/ml-100k/u1.base')
    test = mb.build_interactions_matrix(n_users, n_items, '../data/ml-100k/u1.test')

    model = LightFM(learning_rate=0.01, loss='warp', no_components=30)

    model.fit_partial(train, 
            #   user_features=user_info, 
            #   item_features=item_info,
              epochs=10)

    # train_2 = mb.build_interactions_matrix(n_users, n_items, '../data/ml-100k/u2.base')
    # model.fit_partial(train_2, epochs=10)

    train_precision = precision_at_k(model, 
                                     train, 
                                    #  user_features=user_info, 
                                    #  item_features=item_info,
                                     k=10).mean()

    test_precision = precision_at_k(model, 
                                    test, 
                                    # user_features=user_info, 
                                    # item_features=item_info,
                                    k=10).mean()

    train_auc = auc_score(model, 
                          train, 
                        #   user_features=user_info,
                        #   item_features=item_info
                          ).mean()

    test_auc = auc_score(model, 
                         test, 
                        #  user_features=user_info,
                        #  item_features=item_info
                         ).mean()

    print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
    print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
    print('=================')

    # _, user_repr = model.get_user_representations()
    # _, item_repr = model.get_item_representations()

    # print("begin building annoy {0}, count: {1}".format(time.time(), user_repr.shape[0]))
    # ann = AnnoyIndex(30)
    # for i in xrange(user_repr.shape[0]):
    #     user_embedding = user_repr[i]
    #     ann.add_item(i, user_embedding)
    # ann.build(10)
    # print('finish building annoy {}'.format(time.time()))

    # similar_users = ann.get_nns_by_item(0, 10)
    # print(similar_users)

    # pred = []
    # for user in similar_users:
    #     p = np.matmul(user_repr[user], item_repr.T)
    #     pred.append(p)

    # reduced_mean = reduce((lambda x, y: x + y), pred)

    # print(np.argmax(reduced_mean))
    # top_12 = np.argsort(-reduced_mean)[: 12]
    # print(top_12)

    # user_ids = range(n_users)
    # item_ids = range(n_items)

    # songs = model.predict([0], item_ids)
    # print(songs)
    # print(np.argmax(songs))
