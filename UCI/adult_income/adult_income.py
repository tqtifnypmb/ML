import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import svm
from sklearn import metrics

COLUMNS_NAME = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                'marital-status', 'occupation', 'relationship', 'race', 'gender',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

RACE = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']

DATA_PATH = ('../uci_data/adult.data.txt',
             '../uci_data/adult.test.txt')

def preprocess(data):
    # remove row with missing value
    data = data[data["workclass"] != "?"]
    data = data[data["occupation"] != "?"]
    data = data[data["native-country"] != "?"]

    data.replace(['Married-civ-spouse', 'Divorced', 'Never-married', 
                  'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
                 ['married', 'not married', 'not married', 
                  'not married', 'not married', 'married', 'married'], inplace=True)

    category_col =['race','marital-status', 'gender', 'income', 'relationship']
    for col in category_col:
        _, c = np.unique(data[col], return_inverse=True)
        data[col] = c

    return data

def load_data():
    train_data = pd.read_csv(DATA_PATH[0], header=None, names=COLUMNS_NAME)
    test_data = pd.read_csv(DATA_PATH[1], header=None, names=COLUMNS_NAME)

    # the first row is invalid
    test_data = test_data.drop([0])
    
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)

    return (train_data, test_data)

def naive_bayes_model(x_train, y_train, x_test, y_test):
    model = naive_bayes.GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    acc = metrics.accuracy_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    print('GaussianNB acc: {0}, auc:{1}'.format(acc, auc))

def logistic_regress_model(x_train, y_train, x_test, y_test):
    model = linear_model.LogisticRegressionCV()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    print('Logistic acc: {0}, auc:{1}'.format(acc, auc))

def svm_model(train_data, test_data):
    model = svm.SVC()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    print('SVC acc: {0}, auc:{1}'.format(acc, auc))

if __name__ == '__main__':
    train_data, test_data = load_data()
    
    feature_cols = ['marital-status', 'education-num', 'relationship', 'age']
    y_train = train_data['income']
    x_train = train_data[feature_cols]
    y_test = test_data['income']
    x_test = test_data[feature_cols]

    naive_bayes_model(x_train, y_train, x_test, y_test)
    logistic_regress_model(x_train, y_train, x_test, y_test)

    #extremely slow
    #svm_model(train_data, test_data)