import numpy as np
import time

from math import exp
from sklearn import metrics
from random import randint

# Condider single hidden layer only 
# bias are ignored
class MLP:
    def __init__(self, hidden_neuron_num, learning_rate = 0.01):
        self.hidden_neuron_num_ = hidden_neuron_num
        self.rd = np.random.RandomState(seed=123456)
        self.learning_rate_ = learning_rate

    def _relu(self, x):
        return max(0, x)

    def _sigmoid(self, x):
        return exp(x) / (1 + exp(x))

    def _forward(self, features):
        # input layer --> hidden layer
        for i in range(self.hidden_neuron_num_):
            self.hidden_values[i] = self._relu(np.matmul(features, self.input_weight))

        # hidden layer --> output layer
        output_value = np.matmul(np.transpose(self.hidden_weight), self.hidden_values)
        return self._sigmoid(output_value[0][0])

    def _backward(self, features, pred, label):
        # output layer --> hidden layer
        output_phi = (pred - label) * pred * (1.0 - pred)
        hidden_phis = np.zeros((self.hidden_neuron_num_, 1))

        for i in range(self.hidden_neuron_num_):
            oi = self.hidden_values[i]
            hidden_phis[i] = self.hidden_weight[i] * output_phi * oi * (1.0 - oi)
            gradient = output_phi * oi
            self.hidden_weight[i] -= self.learning_rate_ * gradient

        # hidden layer --> input layer
        n_features = self.samples.shape[1]
        for i in range(n_features):
            oi = features[i]
            gradient = 0
            for j in range(self.hidden_neuron_num_):
                gradient += oi * hidden_phis[j]

            self.input_weight[i] -= gradient * self.learning_rate_
        
    def _is_convergent(self, output, y):
        diff = abs(output - y)
        return diff < 0.01

    def _train(self):
        n_sample = self.samples.shape[0]

        while True:
            epoch = randint(0, n_sample - 1)
    
            features = self.samples[epoch]
            label = self.labels[epoch][0]

            pred = self._forward(features)

            if self._is_convergent(pred, label):
                break

            self._backward(features, pred, label)

    def fit(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError('the dimension of x is not compatible with y')

        n_features = x.shape[1]
        n_units = self.hidden_neuron_num_

        self.samples = x
        self.labels = y

        self.input_weight = self.rd.rand(n_features, 1)
        self.hidden_weight = self.rd.rand(n_units, 1)
        self.hidden_values = np.zeros((n_units, 1))

        self._train()

    def predict(self, x):
        n_smaples = x.shape[0]
        labels = np.zeros((n_smaples, 1))
        for i in range(n_smaples):
            labels[i] = self._forward(x[i])
        
        return labels


def cal_metrics(y_true, y_pred):
    acc = metrics.accuracy_score(y_true, y_pred)
    print("acc {0}".format(acc))

    prec = metrics.precision_score(y_true, y_pred, average='weighted')
    print("precision {0}".format(prec))

    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    print("recall {0}".format(recall))

if __name__ == '__main__':
    mlp = MLP(3)

    x = np.arange(0, 100).reshape((10, -1))
    y = np.arange(0,10).reshape(10, -1)
    mlp.fit(x, y)

    y_pred = mlp.predict(x)
    # cal_metrics(y, y_pred)