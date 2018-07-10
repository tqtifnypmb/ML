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
        self.learning_rate_ = learning_rate
        self.round_ = 500
        self.cur_round_ = 0

    def _relu(self, x):
        return max(0, x)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def _sigmoid(self, x):
        if x < 0:
            return 1 - 1/(1 + exp(x))
        else:
            return 1/(1 + exp(-x))

    def _forward(self, input):
        features = input.reshape(1, -1)

        # input layer --> hidden layer
        for i in range(self.hidden_neuron_num_):
            z = np.matmul(features, self.input_weight[i].T)
            self.hidden_values[i] = self._sigmoid(z)

            for j in range(features.shape[1]):
                x = features[0][j]
                self.gradients[i][j] = self._sigmoid(x) * (1 - self._sigmoid(x))

        # hidden layer --> output layer
        n_labels = self.labels.shape[0]
        for i in range(n_labels):
            z = np.matmul(self.hidden_values.reshape(1, -1), self.hidden_weight[i].reshape(1, -1).T)
            self.output_value[i] = z
        
        return self._softmax(self.output_value)

    def _backward(self, epoch, input, pred, label):
        features = input.reshape(1, -1)

        # output layer --> hidden layer
        n_labels = self.labels.shape[0]
        n_features = features.shape[1]
        hidden_gradients = np.zeros((n_labels, 1))

        for i in range(n_labels):
            p = 1 if label == i else 0
            hidden_gradients[i] = pred[i] - p

        total_hidden_gradient = np.sum(hidden_gradients)
        self.hidden_weight -= total_hidden_gradient * self.learning_rate_

        # hidden layer --> input layer
        for i in range(self.hidden_neuron_num_):
            for j in range(n_features):
                gradient = self.gradients[i][j] * total_hidden_gradient
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

            self._backward(epoch, features, pred, label)

    def fit(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError('the dimension of x is not compatible with y')

        n_labels = y.shape[0]
        n_features = x.shape[1]
        n_units = self.hidden_neuron_num_

        self.samples = x
        self.labels = y

        self.input_weight = np.random.randn(n_units, n_features)
        self.hidden_weight = np.random.randn(n_labels, n_units)
        
        self.gradients = np.zeros((n_units, n_features))
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
    print(np.argmax(y_pred))
    # cal_metrics(y, y_pred)