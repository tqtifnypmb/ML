from sklearn import linear_model
from sklearn import datasets
from sklearn import pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

def generate_data():
    xs = np.linspace(0, 10, 100)
    np.random.RandomState(0).shuffle(xs)
    #xs = np.sort(xs[:20])
    ys = np.sin(xs)
    x = xs[:,np.newaxis]
    y = ys[:, np.newaxis]
    
    return (x, y)

def fit_normal(train_x, train_y, degree):
    regr = pipeline.make_pipeline(PolynomialFeatures(degree), linear_model.LinearRegression())
    regr.fit(train_x, train_y)
    return regr

def fit_lasso(train_x, train_y, degree):
    regr = pipeline.make_pipeline(PolynomialFeatures(degree), linear_model.Lasso())
    regr.fit(train_x, train_y)
    return regr

def fit_ridge(train_x, train_y, degree):
    regr = pipeline.make_pipeline(PolynomialFeatures(degree), linear_model.Ridge())
    regr.fit(train_x, train_y)
    return regr

def predict_and_plot(model, test_x, label):
    pred_y = model.predict(test_x)
    plt.plot(pred_y, label=label)

def draw_figure(num, train_x, train_y, regression_func, title):
    plt.figure(num)
    plt.scatter(train_x[:20], train_y[:20], c='navy', label='training points')

    for c, degree in enumerate([3, 5, 10, 20]):
        model = regression_func(train_x, train_y, degree)
        predict_and_plot(model, test_x, "degree %d" % degree)
    
    plt.title(title)
    plt.yscale('symlog', linthreshy=0.1)
    plt.xlim((0, 20))
    plt.legend(loc='lower right')

if __name__ == '__main__':
    (data_x, data_y) = generate_data()
    train_x = data_x[30:]
    train_y = data_y[30:]
    
    test_x = data_x[0:30]
    test_y = data_y[0:30]

    draw_figure(1, train_x, train_y, fit_lasso, 'Lasso')
    draw_figure(2, train_x, train_y, fit_ridge, 'Ridge')
    draw_figure(3, train_x, train_y, fit_normal, 'No regularization')
    plt.show()
