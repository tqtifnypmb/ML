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
        self.round_ = 200
        self.cur_round_ = 0

    def _relu(self, x):
        return max(0, x)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
        # return np.exp(x) / np.sum(np.exp(x))

    def _sigmoid(self, x):
        if x < 0:
            return 1 - 1/(1 + exp(x))
        else:
            return 1/(1 + exp(-x))

    def _forward(self, features):
        # input layer --> hidden layer
        for i in range(self.hidden_neuron_num_):
            self.hidden_values[i] = self._relu(np.matmul(np.transpose(features), self.input_weight[i]))

        # hidden layer --> output layer
        n_labels = self.labels.shape[0]
        for i in range(n_labels):
            self.output_value[i] = np.matmul(np.transpose(self.hidden_weight[i]), self.hidden_values)
        
        return self._softmax(self.output_value)

    def _backward(self, features, pred, label):
        # output layer --> hidden layer
        n_labels = self.labels.shape[0]
        output_phis = np.zeros((n_labels, 1))
        hidden_phis = np.zeros((self.hidden_neuron_num_, 1))

        for i in range(n_labels):
            p = 1 if label == i else 0
            output_phis[i] = (pred[i] - p) * pred[i] * (1.0 - pred[i])

            for j in range(self.hidden_neuron_num_):
                oi = self.hidden_values[j]
                hidden_phis[j] = self.hidden_weight[i][j] * output_phis[i] * oi * (1.0 - oi)
                gradient = output_phis[i] * oi
                self.hidden_weight[i][j] -= self.learning_rate_ * gradient
        
        # hidden layer --> input layer
        n_features = self.samples.shape[1]
        for i in range(self.hidden_neuron_num_):
            gradient = 0
            for j in range(n_features):
                oi = features[j]
                gradient += oi * hidden_phis[i]
                self.input_weight[i][j] -= gradient * self.learning_rate_
        
    def _is_convergent(self, output, y):
        if self.cur_round_ > self.round_:
            return True
        else:
            self.cur_round_ += 1
            return False

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

        n_labels = y.shape[0]
        n_features = x.shape[1]
        n_units = self.hidden_neuron_num_

        self.samples = x
        self.labels = y

        self.input_weight = self.rd.rand(n_units, n_features)
        self.hidden_weight = self.rd.rand(n_labels, n_units)
       
        self.output_value = np.zeros((n_labels, 1))
        self.hidden_values = np.zeros((n_units, 1))

        self._train()

    def predict(self, x):
        n_smaples = x.shape[0]
        n_labels = self.labels.shape[0]
        labels = np.zeros((n_smaples, 1))
        self.prob_ = np.zeros((n_smaples, n_labels))
        for i in range(n_smaples):
            k = self._forward(x[i])
            self.prob_[i] = k.reshape(1, -1)
            labels[i] = np.argmax(self.prob_[i])
        
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
    
    print(y_pred)
    print(y)
    # print(np.argmax(y_pred))
    # cal_metrics(y, y_pred)