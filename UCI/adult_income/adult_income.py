import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import svm

COLUMNS_NAME = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

RACE = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
MARITAL_STATUS = ['Married-civ-spouse', 'Divorced', 'Never-married', 
                  'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']

DATA_PATH = ('../uci_data/adult.data.txt',
             '../uci_data/adult.test.txt')

def preprocess(data):
    # almost bear no information
    del data['workclass']
    del data['occupation']
    del data['native-country']
    del data['capital-gain']
    del data['capital-loss']

    del data['fnlwgt']

    # redundant
    del data['education']
    del data['relationship']

    def convert_str_to_numerical(column, one):
        origin = data[column]
        new = origin.str.contains(one)
        data[column] = new.astype(int)

    # convert sex feature into numerical 1 - Male, 0 - Female
    convert_str_to_numerical('sex', 'Male')

    # income -> numerical: 1 - <=50k, 0 - >50k
    convert_str_to_numerical('income', '<=50K')

    def map_list_to_numerical(column, lst):
        origin = data[column]
        new = origin.map(lambda x: lst.index(x.strip()))
        data[column] = new

    map_list_to_numerical('race', RACE)
    map_list_to_numerical('marital-status', MARITAL_STATUS)

def load_data():
    train_data = pd.read_csv(DATA_PATH[0], header=None, names=COLUMNS_NAME)
    test_data = pd.read_csv(DATA_PATH[1], header=None, names=COLUMNS_NAME)

    # the first row is invalid
    test_data = test_data.drop([0])
    
    preprocess(train_data)
    preprocess(test_data)

    return (train_data, test_data)

def naive_bayes_model(x_train, y_train, x_test, y_test):
    model = naive_bayes.GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    miss = (y_pred != y_test).astype(int).sum()
    print('GaussianNB error rate: %f' % (miss / len(y_test) * 100))

def logistic_regress_model(x_train, y_train, x_test, y_test):
    model = linear_model.LogisticRegressionCV()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    miss = (y_pred != y_test).astype(int).sum()
    print('Logistic error rate: %f' % (miss / len(y_test) * 100))

def svm_model(train_data, test_data):
    model = svm.SVR()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    miss = (y_pred != y_test).astype(int).sum()
    print('SVC error rate: %f' % (miss / len(y_test) * 100))

if __name__ == '__main__':
    train_data, test_data = load_data()
    
    print(train_data)
    
    y_train = train_data['income']
    x_train = train_data.drop('income', axis=1)
    y_test = test_data['income']
    x_test = test_data.drop('income', axis=1)

    naive_bayes_model(x_train, y_train, x_test, y_test)
    logistic_regress_model(x_train, y_train, x_test, y_test)
    #svm_model(train_data, test_data)