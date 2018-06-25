import numpy as np
import time
from math import exp

class MLP:
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes_ = hidden_layer_sizes
        self.rd = np.random.RandomState(seed=123456)

    def _relu(self, x):
        return max(0, x)

    def _sigmoid(self, x):
        return exp(x) / (1 + exp(x))

    def _forward(self, features, label):
        n_units = self.hidden_layer_sizes_[0]
        
        for i in range(n_units):
            w = self.weights[i]
            self.hidden_values[i] = np.matmul(features, w)

        sv = np.apply_along_axis(self._relu, 1, self.hidden_values)
        output_value = np.matmul(np.transpose(self.output_weights), sv)
        
        return self._sigmoid(output_value[0][0])

    def _backward(self):
        pass

    def _is_convergent(self, output, y):
        return False

    def _train(self):
        n_sample = self.samples.shape[0]

        for epoch in range(n_sample):
            features = self.samples[epoch]
            label = self.labels[epoch]

            output = self._forward(features, label)
            
            if self._is_convergent(output, label):
                break

            self._backward()

    def fit(self, x, y):
        if x.shape[0] != y.shape[0]:
            raise ValueError('the dimension of x is not compatible with y')

        n_samples = x.shape[0]
        n_features = x.shape[1]
        n_units = self.hidden_layer_sizes_[0]

        one = np.ones((n_samples, 1))
        self.samples = np.hstack((x, one))
        self.labels = y

        self.weights = self.rd.rand(n_units, n_features + 1)

        self.hidden_values = np.zeros((n_units + 1, 1))
        self.output_weights = self.rd.rand(n_units + 1, 1)

        self._train()

    def predict(self, x):
        pass

if __name__ == '__main__':
    mlp = MLP((3,1))

    x = np.arange(0, 100).reshape((10, -1))
    y = np.arange(0,10).reshape(10, -1)
    y_pred = mlp.fit(x, y)
