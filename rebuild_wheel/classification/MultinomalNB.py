import numpy as np
from sklearn import naive_bayes
from utility import group_features, test_model

class MultinomialNaiveBayes:
    def __init__(self, alpha = 1.0):
        # alpha for Laplace smooth
        self.alpha_ = alpha

    def fit(self, X, Y):
        classes = np.unique(Y)
        n_classes = Y.shape[0]
        
        # cal log[ p(y) ]
        self.class_log_prior_ = np.zeros((classes.shape[0], 1))
        for kls in Y:
            self.class_log_prior_[kls] += 1

        self.class_log_prior_ /= n_classes
        self.class_log_prior_ = np.log(self.class_log_prior_)

        # cal p(x | y)
        n_features = X.shape[1]
        class_idx = group_features(X, Y)
        self.feature_log_prob_ = np.zeros((len(classes), n_features))
        for kls in classes:
            indices = class_idx[kls]
            samples = X[indices]
            smoothed_fc = np.sum(samples, axis=0) + self.alpha_
            smoothed_cc = np.sum(smoothed_fc)
            smoothed = np.log(smoothed_fc) - np.log(smoothed_cc)
            self.feature_log_prob_[kls:] = smoothed

    def predict(self, X):
        likelihood = np.dot(X, self.feature_log_prob_.T) + self.class_log_prior_.reshape(-1,3)
        return np.argmax(likelihood, 1)

if __name__ == '__main__':
    test_model(naive_bayes.MultinomialNB(), MultinomialNaiveBayes())