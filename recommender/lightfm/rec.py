from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

from matrix_builder import UserItemMatrix 

import pandas as pd
import numpy as np

    
if __name__ == '__main__':
    
    mb = UserItemMatrix()
    
    user_info = mb.build_user_info_matrix('../data/ml-100k/u.user')
    item_info = mb.build_item_info_matrix('../data/ml-100k/u.item')

    n_users = user_info.shape[0]
    n_items = item_info.shape[0]

    print(user_info.shape)
    train = mb.build_interactions_matrix(n_users, n_items, '../data/ml-100k/u1.base')

    test = mb.build_interactions_matrix(n_users, n_items, '../data/ml-100k/u1.test')

    model = LightFM(learning_rate=0.05, loss='bpr', no_components=30)
    # model.fit(train, epochs=10)
  
    # train_precision = precision_at_k(model, train, k=10).mean()
    # test_precision = precision_at_k(model, test, k=10).mean()

    # train_auc = auc_score(model, train).mean()
    # test_auc = auc_score(model, test).mean()

    # print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
    # print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
    # print('=================')

    model.fit(train, user_features=user_info, epochs=10)
    train_precision = precision_at_k(model, train, k=10).mean()
    test_precision = precision_at_k(model, test, k=10).mean()

    train_auc = auc_score(model, train).mean()
    test_auc = auc_score(model, test).mean()

    print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
    print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
    print('=================')