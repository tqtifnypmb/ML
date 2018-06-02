import pandas as pd
import numpy as np
from sklearn import feature_selection as fs

COLUMN_NAMES = ['sex', 'length', 'diameter', 'height', 
                'whole weight', 'shucked weight', 'viscera weight',
                'shell weight', 'rings']

# feature selection

def cal_features_mutual_info(data):
    y = data['rings']
    features = data.loc[:, data.columns != 'rings']

    info = fs.mutual_info_regression(features, y)
    
    print('========== mutual info ==============')
    for idx, col in enumerate(COLUMN_NAMES):
        if col == 'rings':
            break
        name = COLUMN_NAMES[idx]
        print('{0} ==> {1}'.format(name, info[idx]))
    print('========================')

def cal_feature_variance(data):
    vt = fs.VarianceThreshold()
    vt.fit_transform(data)
    print('======== variance ================')
    for idx, col in enumerate(COLUMN_NAMES):
        print('{0} ==> {1}'.format(col, vt.variances_[idx]))
    print('========================')
    
# data loading / preprocessing

def preprocessing(data):
    _, v = np.unique(data['sex'], return_inverse=True)
    data['sex'] = v

def load_data():
    data = pd.read_csv('../uci_data/abalone.data.txt', header=None, names=COLUMN_NAMES)
    preprocessing(data)
    return data

if __name__ == '__main__':
    data = load_data()
    cal_features_mutual_info(data)
    cal_feature_variance(data)