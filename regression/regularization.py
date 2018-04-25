from sklearn import linear_model
from sklearn import datasets
from sklearn import pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

def generate_data():
    xs = np.linspace(0, 10, 1000)
    np.random.RandomState(0).shuffle(xs)
    #xs = np.sort(xs)
    ys = np.sin(xs)
    x = xs[:,np.newaxis]
    y = ys[:, np.newaxis]
    
    return (x, y)

def fit_normal(train_x, train_y, degree):
    regr = pipeline.make_pipeline(PolynomialFeatures(degree), linear_model.LinearRegression())
    regr.fit(train_x, train_y)
    return regr

def fit_lasso(train_x, train_y, degree):
    regr = pipeline.make_pipeline(PolynomialFeatures(degree), linear_model.LassoCV())
    regr.fit(train_x, train_y)
    return regr

def fit_ridge(train_x, train_y, degree):
    regr = pipeline.make_pipeline(PolynomialFeatures(degree), linear_model.RidgeCV())
    regr.fit(train_x, train_y)
    return regr

def predict_and_plot(model, test_x, label):
    pred_y = model.predict(test_x)
    plt.plot(pred_y, label=label)

def draw_figure(num, data_x, data_y, regression_func, title):
    train_x = data_x[30:]
    train_y = data_y[30:]
    
    test_x = data_x[0:30]
    test_y = data_y[0:30]

    plt.figure(num)
    plt.scatter(test_x, test_y, c='navy', label='testing points')

    for c, degree in enumerate([1, 3, 5, 15]):
        model = regression_func(train_x, train_y, degree)
        predict_and_plot(model, test_x, "degree %d" % degree)
    
    plt.title(title)
    plt.yscale('symlog', linthreshy=0.1)
    plt.xlim((0, 20))
    plt.legend(loc='lower right')

if __name__ == '__main__':
    (data_x, data_y) = generate_data()
    
    draw_figure(1, data_x, data_y, fit_lasso, 'Lasso')
    draw_figure(2, data_x, data_y, fit_ridge, 'Ridge')
    draw_figure(3, data_x, data_y, fit_normal, 'No regularization')
    plt.show()
