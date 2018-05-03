from scipy import linalg
from sklearn import datasets
from sklearn import linear_model
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np

class LinearRegressor:
    def fit(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError('incompatible dimensions')

        # Linear regression can be regarded as projecting
        # y onto the space spanned by design matrix.
        w, _, _, _ = linalg.lstsq(x, y)
        self.weights = w

    def predit(self, x):
        return np.dot(x, self.weights)

class RidgeRegressor:
    def __init__(self, alpha = 2):
        self.alpha = alpha

    def fit(self, x, y):
        self.regressor = linear_model.SGDRegressor(alpha=self.alpha)
        self.regressor.fit(x, y)

    def predict(self, x):
        return self.regressor.predict(x)

if __name__ == '__main__':
    data = datasets.load_diabetes()
    train_x, test_x, train_y, test_y = model_selection.train_test_split(data.data, data.target, test_size=0.3)
    
    _, axs = plt.subplots(3, 1)

    custom_regr = LinearRegressor()
    custom_regr.fit(train_x, train_y)
    custom_y_pred = custom_regr.predit(test_x)
    axs[0].plot(custom_y_pred, label='custom regressor')
    axs[0].legend(loc='lower right')

    regr = linear_model.LinearRegression()
    regr.fit(train_x, train_y)
    y_pred = regr.predict(test_x)
    axs[1].plot(y_pred, label='regressor')
    axs[1].legend(loc='lower right')

    ridge = RidgeRegressor()
    ridge.fit(train_x, train_y)
    y_pred_2 = ridge.predict(test_x)
    axs[2].plot(y_pred_2, label='ridge regressor')
    axs[2].legend(loc='lower right')
    plt.show()