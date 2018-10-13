import numpy as np

# Linear regression has closed-form solution
class LinearRegression:
    def fit(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError('incompatible dimensions')

        # Linear regression can be regarded as projecting
        # y onto the space spanned by design matrix.
        w, _, _, _ = np.linalg.lstsq(x, y, rcond=-1)
        self.weights = w

    def predit(self, x):
        return np.dot(x, self.weights)

class LinearRegression2:
    def __init__(self, num_features, lr = 1e-2):
        # FIXME: weight initialization
        self.weights = np.zeros((1, num_features))
        self.lr = lr

    def fit(self, x, y):
        if x.shape[1] != self.weights.shape[1] or x.shape[0] != y.shape[0]:
            raise ValueError('incompatible dimensions')

        # train by stochastic gradient descent
        for row_idx in range(len(x)):
            sample = x[row_idx]
            label = y[row_idx]
        
            h = np.dot(self.weights, sample)[0]
            e = label - h
            delta = self.lr * e

            for feature_idx in range(len(self.weights[0])):
                old_w = self.weights[0][feature_idx]
                feature = x[row_idx][feature_idx]
                new_w = old_w + delta * feature
                self.weights[0][feature_idx] = new_w

    def predict(self, x):
        y = np.zeros([x.shape[0], 1])
        for row_idx in range(len(x)):
            sample = x[row_idx]
            v = np.dot(self.weights, sample)[0]
            y[row_idx] = v
        return y