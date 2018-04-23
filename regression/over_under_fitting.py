from sklearn import linear_model
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import itertools

# generate linear data with gaussian noise ~ N(e|0, 0)
def generate_data():
    data = datasets.make_regression(100, 100, coef=True)
    return data

def feature_map(train_data_x, basis_function):
    row_idx = 0
    for row in train_data_x:
        col_idx = 1
        for value in row:
            train_data_x[row_idx][col_idx - 1] = basis_function(value, col_idx)
            col_idx += 1
        row_idx += 1
    return train_data_x

def regression(train_data_x, train_data_y):
    train_data_x = np.copy(train_data_x)
    regr = linear_model.LinearRegression(copy_X=True)
    regr.fit(train_data_x, train_data_y)
    return regr
    
def overfitting_regression(train_data_x, train_data_y, regularization = False):
    train_data_x = np.copy(train_data_x)
    polynomial_basis = lambda x, idx: x ** idx
    train_data_x = feature_map(train_data_x, polynomial_basis)

    regr = None
    if regularization:
        regr = linear_model.Ridge(copy_X=True)
    else:
        regr = linear_model.LinearRegression(copy_X=True)
    regr.fit(train_data_x, train_data_y)
    return regr

def underfitting_regression(train_data_x, train_data_y):
    train_data_x = np.copy(train_data_x)
    train_data_x[:,9] = 0
    regr = linear_model.LinearRegression(copy_X=True)
    regr.fit(train_data_x, train_data_y)
    return regr

def plot(model, test_data_x):
    pred_y = model.predict(test_data_x)
    plt.plot(pred_y)

    
if __name__ == '__main__':
    train_data = generate_data()
    train_data_x = train_data[0]
    train_data_y = train_data[1]
    coef = train_data[2]
    
    test_data_x = train_data_x[0:10]
    test_data_y = train_data_y[0:10]
    train_data_x = train_data_x[10:]
    train_data_y = train_data_y[10:]

    plt.plot(test_data_y)

    regr = regression(train_data_x, train_data_y)
    plot(regr, test_data_x)

    under_regr = underfitting_regression(train_data_x, train_data_y)
    plot(under_regr, test_data_x)

    over_regr = overfitting_regression(train_data_x, train_data_y)
    plot(over_regr, test_data_x)

    r_regr = overfitting_regression(train_data_x, train_data_y, regularization = True)
    plot(r_regr, test_data_x)

    plt.legend(['origin', 'regression', 'under', 'over', 'r-over'], loc='upper left')
    plt.show()