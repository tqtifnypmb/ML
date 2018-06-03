import pandas as pd
import numpy as np
import math

from sklearn import feature_selection as fs
from sklearn import naive_bayes
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model

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

def cal_feature_variance(data):
    vt = fs.VarianceThreshold()
    vt.fit_transform(data)
    print('======== variance ================')
    for idx, col in enumerate(COLUMN_NAMES):
        print('{0} ==> {1}'.format(col, vt.variances_[idx]))
    
# data loading / preprocessing

def preprocessing(data):
    _, v = np.unique(data['sex'], return_inverse=True)
    data['sex'] = v

def load_data():
    data = pd.read_csv('../uci_data/abalone.data.txt', header=None, names=COLUMN_NAMES)
    preprocessing(data)
    return data

# metrics

# 1. metrics for multi-class classification problem
def cal_metrics(y_test, y_pred, label):
    acc = metrics.accuracy_score(y_test, y_pred)
    print('{0} acc: {1}'.format(label, acc))

# models

# 1. imbalance multi-class labels
def gaussian_naive_bayes(x_train, y_train, x_test, y_test):
    model = naive_bayes.GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cal_metrics(y_test, y_pred, 'gaussianNB')

def multinomial_naive_bayes(x_train, y_train, x_test, y_test):
    model = naive_bayes.MultinomialNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cal_metrics(y_test, y_pred, 'multinomialNB')

def logistics_regression(x_train, y_train, x_test, y_test):
    model = linear_model.LogisticRegression(solver='sag', multi_class='multinomial')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cal_metrics(y_test, y_pred, 'logisticsc regression')

if __name__ == '__main__':
    data = load_data()
    cal_features_mutual_info(data)
    cal_feature_variance(data)
    print('==================')

    # ignore features with low variance
    selected_features = ['sex', 'length', 'whole weight', 
                         'shucked weight', 'viscera weight',
                         'shell weight']

    split_point = math.floor(len(data) * 0.8) 
    data_train = data[: split_point]
    data_test = data[split_point:]

    x_train = data_train[selected_features]
    y_train = data_train['rings']
    x_test = data_test[selected_features]
    y_test = data_test['rings']
    gaussian_naive_bayes(x_train, y_train, x_test, y_test)
    logistics_regression(x_train, y_train, x_test, y_test)
    multinomial_naive_bayes(x_train, y_train, x_test, y_test)