import numpy as np

class LogisticsRegression:
    def __init__(self, num_features, lr = 1e-2):
        # FIXME: weight initialization
        self.weights = np.random.normal(size=(1, num_features))
        self.lr = lr

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y):
        if x.shape[1] != self.weights.shape[1] or x.shape[0] != y.shape[0]:
            raise ValueError('incompatible dimensions')

        # train by stochastic gradient descent
        for row_idx in range(len(x)):
            sample = x[row_idx]
            label = y[row_idx]
        
            z = np.dot(self.weights, sample)[0]
            h = self._sigmoid(z)
            e = label - h
            delta = self.lr * e

            # FIXME: gradient vanish ?
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
            v = self._sigmoid(v)
            y[row_idx] = v
        return y

class GaussianNaiveBayesian:
    def __init__(self, prior=None, var_smoothing=1e-9):
        self.prior = prior
        self.var_smoothing = var_smoothing
    
    def fit(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError('incompatible dimensions')

        klass, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
        num_klass = len(klass)
        num_features = x.shape[1]

        feature_sums = np.zeros([num_klass, num_features])
        self.mu = np.zeros_like(feature_sums)
        self.sigma = np.zeros_like(feature_sums)

        # sum of individual feature in each class
        for row_idx in range(len(inverse)):
            index = inverse[row_idx]
            sample = x[row_idx]
            feature_sums[index] += sample

        # means of features
        for klass_idx in range(num_klass):
            u = feature_sums[klass_idx] / counts[klass_idx]
            self.mu[klass_idx] = u
            self.sigma[klass_idx] = ((feature_sums[klass_idx] - u) ** 2) / counts[klass_idx]
        
        self.sigma += self.var_smoothing

    def predict(self, x):
        # P(x1|Y) * p(x2|Y) * ...
        num_klass = self.mu.shape[0]
        num_features = self.mu.shape[1]

        likelihood = np.zeros([x.shape[0], num_klass])
        for klass in range(num_klass):
            klass_mu = self.mu[klass]
            klass_sigma = self.sigma[klass]

            for row_idx in range(x.shape[0]):
                row = x[row_idx]
                sum = 0
                for feature_idx in range(num_features):
                    feature = row[feature_idx]
                    feature_mu = klass_mu[feature_idx]
                    feature_sigma = klass_sigma[feature_idx]

                    n = 1 / 2 * np.log(1 / (2 * np.pi * feature_sigma * feature_sigma))
                    m = -1 / (2 * feature_sigma) * ((feature - feature_mu) ** 2)
                    sum += n + m

                likelihood[row_idx][klass] = sum

        y = np.argmax(likelihood, axis=1)
        return y
        

class MultinormalNaiveBayesian:
    pass