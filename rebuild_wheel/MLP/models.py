import numpy as np

class _Unit:
    def __init__(self, num_units, lr, activation = 'relu'):
        self.weights = np.random.normal(size=(1, num_units)) 
        self.bias = 0
        self.lr = lr

        if activation == 'relu':
            self.activation = self._relu
        elif activation == 'sigmoid':
            self.activation = self._sigmoid
        else:
            raise ValueError('unknown activation')

    def _relu(self, x, derivative=False):
        if not derivative:
            return x if x > 0 else 0
        else:
            return 1 if x > 0 else 0

    def _sigmoid(self, x, derivative=False):
        if not derivative:
            return np.exp(x) / (1 + np.exp(x))
        else:
            return np.exp(x) / ((1 +  np.exp(x)) ** 2)

    def forward(self, x):
        z = np.dot(self.weights, x)[0]
        a = self.activation(z)
        self.derivative_activation = self.activation(z, True)
        self.forward_input = x
        return a

    def backward(self, x):
        d = self.forward_input * self.derivative_activation
        self.weights = self.weights - d * self.lr
        return x * d

class MLP:
    def __init__(self, input_shape, hidden_shape, lr = 1e-2, activation='relu', loss='mse'):
        self.activation = activation
        self._build_layers(lr, input_shape, hidden_shape)

        if loss == 'cross_entropy':
            self.loss = self._cross_entropy
        elif loss == 'mse':
            self.loss = self._mean_squared_error

    def _build_layers(self, lr, input_shape, hidden_shape):
        self.hidden = []

        for layer_idx in range(len(hidden_shape)):
            num_units = hidden_shape[layer_idx]
            prev_layer_size = None

            if layer_idx == 0:
                prev_layer_size = input_shape[1]
            else:
                prev_layer_size = hidden_shape[layer_idx - 1]

            layers = []
            for _ in range(num_units):
                unit = _Unit(prev_layer_size, lr, self.activation)
                layers.append(unit)

            self.hidden.append(layers)

    def _cross_entropy(self, logit, label):
        pass

    def _mean_squared_error(self, logit, label):
        # e = (logit - label) ** 2
        # return np.average(e)
        return logit - label

    def _forward(self, x):
        last_output = x
        for layer in self.hidden:
            num_units = len(layer)
            new_output = np.zeros([num_units])
            
            for idx in range(num_units):
                unit = layer[idx]
                output = unit.forward(last_output)
                new_output[idx] = output

            last_output = new_output
    
        return last_output

    def _backward(self, output):
        last_output = output
        for layer in reversed(self.hidden):
            unit = layer[0]
            input = last_output[0]
            new_output = unit.backward(input)

            remain = layer[1:]
            for idx in range(len(remain)):
                unit = remain[idx]
                input = last_output[idx]
                new_output += unit.backward(input)

            last_output = new_output

    def fit(self, x, y, batch_size=32):
        if x.shape[0] != y.shape[0]:
            raise ValueError('incompatible dimensions')

        batch_loss_value = None
        for data_idx in range(x.shape[0]):
            sample = x[data_idx]
            label = y[data_idx]

            # forward
            last_output = self._forward(sample)
        
            # calculate loss
            for idx in range(len(last_output)):
                logit = last_output[idx]
                loss_value = self.loss(logit, label)
                last_output[idx] = loss_value

            if batch_loss_value is None:
                batch_loss_value = last_output
            else:
                batch_loss_value += last_output

            # backward
            if data_idx % batch_size == 0 and data_idx > 0:
                batch_loss_value /= batch_size
                self._backward(batch_loss_value)
                batch_loss_value = None

        if batch_loss_value is not None:
            count = x.shape[0] % batch_size
            batch_loss_value /= count
            self._backward(batch_loss_value)

    def predict(self, x):
        y = np.zeros((x.shape[0], 3))

        for idx in range(x.shape[0]):
            sample = x[idx]
            pred = self._forward(sample)
            y[idx] = pred
        
        return y