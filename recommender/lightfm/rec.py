from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from lightfm.data import Dataset

import pandas as pd
import numpy as np

item_info_file = None
uesr_info_file = None

def build_item_info_dict():
    pass

def build_user_item_matrix():
    dataset = Dataset()
    
if __name__ == '__main__':
    global item_info_file
    global uesr_info_file

    item_info_file = pd.read_csv('../data/ml-100k/u.item')
    uesr_info_file = pd.read_csv('../data/ml-100k/u.user')

    train = pd.read_csv('../data/ml-100k/ua.base')
    test = pd.read_csv('../data/ml-100k/ua.test')

    model = LightFM(learning_rate=0.05, loss='bpr')
    model.fit(train, epochs=10)

    train_precision = precision_at_k(model, train, k=10).mean()
    test_precision = precision_at_k(model, test, k=10).mean()

    train_auc = auc_score(model, train).mean()
    test_auc = auc_score(model, test).mean()

    print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
    print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))