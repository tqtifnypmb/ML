import numpy as np
import pandas

from sklearn import cluster
from sklearn import preprocessing
from sklearn import feature_selection as fs
from sklearn import model_selection as ms
from sklearn import metrics
from sklearn import naive_bayes
from sklearn import neural_network

DATA_PATH = '../uci_data/car.data.txt'
COLUMN_NAMES = ['buying', 'maint', 'doors', 'persons', 
                'lug_boot', 'safety', 'value']

def load_data():
    data = pandas.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)
    preprocess(data)
    
    print(data.describe())
    return data

def preprocess(data):
    # buying
    le = preprocessing.LabelEncoder()
    le.fit(['vhigh', 'high', 'med', 'low'])
    data['buying'] = le.transform(data['buying'])

    # maint
    data['maint'] = le.transform(data['maint'])

    # lug boot
    le.fit(['small', 'med', 'big'])
    data['lug_boot'] = le.transform(data['lug_boot'])

    # safety
    le.fit(['low', 'med', 'high'])
    data['safety'] = le.transform(data['safety'])

    # value
    le.fit(['unacc', 'acc', 'good', 'vgood'])
    data['value'] = le.transform(data['value'])

    # doors
    data['doors'] = data['doors'].apply(lambda d: 5 if d == '5more' else d)

    # persons
    data['persons'] = data['persons'].apply(lambda d: 5 if d == 'more' else d)

def cal_mutual_information(data):
    Y = data['value']
    features = data.loc[:, data.columns != 'value']

    info = fs.mutual_info_regression(features, Y)
    print('========== mutual info ==============')
    for idx, col in enumerate(COLUMN_NAMES):
        if col == 'value':
            continue
        name = COLUMN_NAMES[idx]
        print('{0} ==> {1}'.format(name, info[idx]))

def cal_variance(data):
    vt = fs.VarianceThreshold()
    vt.fit_transform(data)
    print('======== variance ================')
    for idx, col in enumerate(COLUMN_NAMES):
        print('{0} ==> {1}'.format(col, vt.variances_[idx]))

def feature_selection(data):
    cal_mutual_information(data)
    cal_variance(data)

def cal_metrics(label, y_true, y_pred):
    print('========== {0} ================'.format(label))
    # cross_entropy = metrics.log_loss(y_true, y_pred)
    # print('cross entropy %f' % cross_entropy)

    acc = metrics.accuracy_score(y_true, y_pred)
    print("acc {0}".format(acc))

    prec = metrics.precision_score(y_true, y_pred, average='weighted')
    print("precision {0}".format(prec))

    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    print("recall {0}".format(recall))

def k_means_model(x_train, y_train, x_test, y_test):
    model = cluster.KMeans()
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    cal_metrics('KNN', y_test, y_pred)

def gaussianNB_model(x_train, y_train, x_test, y_test):
    model = naive_bayes.GaussianNB()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    cal_metrics('GaussianNB', y_test, y_pred)

def multinomialNB_model(x_train, y_train, x_test, y_test):
    model = naive_bayes.MultinomialNB()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    cal_metrics('MultinomialNB', y_test, y_pred)

def mlp_model(x_train, y_train, x_test, y_test):
    mlp = neural_network.MLPClassifier(hidden_layer_sizes=(10000,))
    mlp.fit(x_train, y_train)

    y_pred = mlp.predict(x_test)
    cal_metrics('MLP', y_test, y_pred)

if __name__ == '__main__':
    data = load_data()
    
    feature_selection(data)

    Y = data['value']
    X = data.loc[:, data.columns != 'value']
    x_train, x_test, y_train, y_test = ms.train_test_split(X, Y,  test_size=0.3)
    k_means_model(x_train, y_train, x_test, y_test)
    gaussianNB_model(x_train, y_train, x_test, y_test)
    multinomialNB_model(x_train, y_train, x_test, y_test)
    mlp_model(x_train, y_train, x_test, y_test)