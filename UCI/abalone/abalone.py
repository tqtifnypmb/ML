import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn import feature_selection as fs
from sklearn import naive_bayes
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model
from sklearn import svm

from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.over_sampling import SMOTE, RandomOverSampler

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
    
def draw_class_hist(Y):
    bins = [x for x in range(1, 29, 5)]
    Y.plot.hist(bins=bins)
    plt.show()

# data loading / preprocessing

def preprocessing(data):
    _, v = np.unique(data['sex'], return_inverse=True)
    data['sex'] = v

def load_data():
    data = pd.read_csv('../uci_data/abalone.data.txt', header=None, names=COLUMN_NAMES)
    preprocessing(data)
    print(data.describe())
    return data

def oversampling(X, Y):
    # some class has only one sample
    # to apply SMOTE we first oversample it randomly
    X_resampled, Y_resampled = RandomOverSampler().fit_sample(X, Y)
    X_resampled, Y_resampled = SMOTE().fit_sample(X_resampled, Y_resampled)
    return (X_resampled, Y_resampled)

def undersampling(X, Y):
    rus = NeighbourhoodCleaningRule(ratio='majority')
    x_new, y_new = rus.fit_sample(X, Y)
    return (x_new, y_new)

# metrics

# 1. metrics for multi-class classification problem
def cal_metrics(y_test, y_pred, label):
    acc = metrics.accuracy_score(y_test, y_pred)
    print('{0} acc: {1}'.format(label, acc))

    prec = metrics.precision_score(y_test, y_pred, average='weighted')
    print('{0} precision: {1}'.format(label, prec))

    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    print('{0} recall: {1}'.format(label, recall))

# models

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

def select_features_by_stat_info(data):
    cal_features_mutual_info(data)
    cal_feature_variance(data)
    print('==================')

    # ignore features with low variance
    return['sex', 'length', 'whole weight',               
           'shucked weight', 'viscera weight',              
           'shell weight']

def select_feature_by_L1(data_train, data_test):
    all_cols = ['sex', 'length', 'diameter', 'height', 
                'whole weight', 'shucked weight', 'viscera weight',
                'shell weight']
    Y = data_train['rings']
    X = data_train[all_cols]
    X_test = data_test[all_cols]

    svc = svm.LinearSVC(penalty='l1', dual=False).fit(X, Y)
    model = fs.SelectFromModel(svc, threshold=0.5, prefit=True)
    return (model.transform(X), model.transform(X_test))

if __name__ == '__main__':
    data = load_data()
    
    split_point = math.floor(len(data) * 0.8) 
    data_train = data[: split_point]
    data_test = data[split_point:]

    y_train = data_train['rings']
    y_test = data_test['rings']

    print('======== select features by stat info ========')
    selected_features = select_features_by_stat_info(data)
    x_train = data_train[selected_features]
    x_test = data_test[selected_features]
    
    gaussian_naive_bayes(x_train, y_train, x_test, y_test)
    logistics_regression(x_train, y_train, x_test, y_test)
    multinomial_naive_bayes(x_train, y_train, x_test, y_test)

    print('=========== select features by L1 =============')
    x_train, x_test = select_feature_by_L1(data_train, data_test)
    gaussian_naive_bayes(x_train, y_train, x_test, y_test)
    logistics_regression(x_train, y_train, x_test, y_test)
    multinomial_naive_bayes(x_train, y_train, x_test, y_test)

    print('============ under sampling ==============')
    x_res, y_res = undersampling(x_train, y_train)
    gaussian_naive_bayes(x_res, y_res, x_test, y_test)
    logistics_regression(x_res, y_res, x_test, y_test)
    multinomial_naive_bayes(x_res, y_res, x_test, y_test)

    print('============ over sampling ==============')
    x_res, y_res = oversampling(x_train, y_train)
    gaussian_naive_bayes(x_res, y_res, x_test, y_test)
    logistics_regression(x_res, y_res, x_test, y_test)
    multinomial_naive_bayes(x_res, y_res, x_test, y_test)

    #draw_class_hist(data['rings'])