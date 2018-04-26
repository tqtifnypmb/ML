from scipy import linalg
from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

class LinearRegressor:
    def fit(self, x, y):
        """
        Linear regression can be regard as projecting
        y onto the space spanned by design matrix.
        """

        if x.shape[1] != y.shape[0]:
            raise ValueError('incompatible dimensions')

        w, _, _, _ = linalg.lstsq(x, y)
        self.weights = w

    def predit(self, x):
        return np.dot(x, self.weights)

class LassoRegressor:
    pass

if __name__ == '__main__':
    data = datasets.make_regression(100, 100, coef=True)
    train_x = data[0]
    train_y = data[1]
    
    custom_regr = LinearRegressor()
    custom_regr.fit(train_x, train_y)
    custom_y_pred = custom_regr.predit(train_x)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(custom_y_pred, label='custom regressor')
    axs[0].legend(loc='lower right')

    regr = linear_model.LinearRegression()
    regr.fit(train_x, train_y)
    y_pred = regr.predict(train_x)
    axs[1].plot(y_pred, label='regressor')
    axs[1].legend(loc='lower right')
    plt.show()