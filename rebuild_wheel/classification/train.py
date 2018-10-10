import numpy as np

from models import LogisticsRegression, GaussianNaiveBayesian
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics

if __name__ == '__main__':
    iris = datasets.load_iris()
    train_x, test_x, train_y, test_y = model_selection.train_test_split(iris.data, iris.target, test_size=0.4)

    regressor2 = LogisticsRegression(train_x.shape[1])
    regressor2.fit(train_x, train_y)
    pred_y = regressor2.predict(test_x)
    acc = metrics.mean_squared_error(test_y, pred_y)
    print(acc)

    gausianNB = GaussianNaiveBayesian()
    gausianNB.fit(train_x, train_y)
    pred_y = gausianNB.predict(test_x)
    print(pred_y)
    acc = metrics.mean_squared_error(test_y, pred_y)
    print(acc)