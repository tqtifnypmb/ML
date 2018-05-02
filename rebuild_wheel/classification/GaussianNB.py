import numpy as np
from sklearn import naive_bayes
from utility import group_features, test_model

class GaussianNaiveBayes:
    def __init__(self, prior = None, var_smooth = 0.1):
        # in case var of features too small
        self.var_smooth_ = var_smooth
        self.prior_ = prior

    def fit(self, x, y):
        # collect samples belong to the same y
        self.class_ = np.unique(y)
        class_idx = group_features(x, y)

        # cal mean and variate for every features
        n_classes = len(self.class_)
        n_features = x.shape[1]
        self.mu_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        for kls in self.class_:
            indices = class_idx[kls]
            samples = x[indices]
            mu = np.average(samples, axis=0)
            self.mu_[kls:] = mu

            sigma = np.average((samples - mu) ** 2, axis=0)
            self.sigma_[kls:] = sigma

        #self.sigma_ += self.var_smooth_
        

    def predict(self, x):
        # cal log[ p(x1, x2, x3, ... | c) ]
        # assume that p(c1) == p(c2) == p(c3) ...
        likelihood = np.zeros((len(self.class_), x.shape[0]))
        for kls in self.class_:
            n = -0.5 * np.sum(np.log(2 * np.pi * self.sigma_[kls]))
            m = 0.5 * np.sum(((x - self.mu_[kls]) ** 2) / self.sigma_[kls], 1)
            prior = 0
            if self.prior_ is not None:
                prior = np.log(self.prior_[kls])

            likelihood[kls:] = n - m + prior

        return np.argmax(likelihood, 0)


if __name__ == '__main__':
    test_model(naive_bayes.GaussianNB(), GaussianNaiveBayes())