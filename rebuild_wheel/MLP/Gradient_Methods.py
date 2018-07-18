import tensorflow as tf
import numpy as np

from sklearn import datasets
from sklearn import metrics
from sklearn import neural_network
from sklearn import preprocessing
from math import floor, sqrt

class MLP:
    def __init__(self, 
                 learning_rate,
                 gradient_type, 
                 relu = None,
                 batch_size = 100, 
                 max_iteration = 100000, 
                 tolerent = 1e-2,
                 rigid = 1e-6):
        self.learning_rate_ = learning_rate
        self.batch_size_ = batch_size
        self.max_iteration_ = max_iteration
        self.tolerent_ = tolerent
        self.gradient_type_ = gradient_type
        self.relu_ = relu
        self.best_loss_value_ = 1000000.
        self.no_improvement_num_ = 0
        self.max_no_improvement_num_ = 500
        self.rigid_ = rigid

    def _init_weights(self, hidden_shape, output_shape):
        w1 = tf.random_normal(hidden_shape) * sqrt(2.0 / hidden_shape[0])
        w2 = tf.random_normal(output_shape) * sqrt(2.0 / output_shape[0])
        return w1, w2

    def _predifined_optimizer(self, y_pred, y):
        loss = tf.losses.mean_squared_error(y_pred, y)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_)
        optimize_action = optimizer.minimize(loss)

        return loss, optimize_action

    def _vanilia_gradient_descent(self, loss, w1, w2):
        grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])
        new_w1 = w1.assign(w1 - grad_w1 / self.batch_size_ * self.learning_rate_)
        new_w2 = w2.assign(w2 - grad_w2 / self.batch_size_ * self.learning_rate_)

        return loss, tf.group(new_w1, new_w2)

    def _momentum_gradient_descent(self, loss, w1, w2):
        grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

        vel_1_shape = (self.n_features_, self.neuron_num_)
        vel_2_shape = (self.neuron_num_, self.n_labels_dim_)

        velocity1 = tf.Variable(tf.zeros(vel_1_shape))
        velocity2 = tf.Variable(tf.zeros(vel_2_shape))

        gama = 0.9

        velocity1 = velocity1.assign(gama * velocity1 + grad_w1 / self.batch_size_ * self.learning_rate_)
        velocity2 = velocity2.assign(gama * velocity2 + grad_w2 / self.batch_size_ * self.learning_rate_)
        new_w1 = w1.assign(w1 - velocity1)
        new_w2 = w2.assign(w2 - velocity2)
        return loss, tf.group(new_w1, new_w2)

    def _nesterov_gradient_descent(self, input, label, w1, w2): 
        vel_1_shape = (self.n_features_, self.neuron_num_)
        vel_2_shape = (self.neuron_num_, self.n_labels_dim_)
        velocity1 = tf.Variable(tf.zeros(vel_1_shape))
        velocity2 = tf.Variable(tf.zeros(vel_2_shape))
        future_1 = tf.Variable(w1)
        future_2 = tf.Variable(w2)
        gama = 0.9

        hidden = self._relu(input, future_1)
        y_pred = tf.matmul(hidden, future_2)
        diff = y_pred - label
        loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))

        grad_1, grad_2 = tf.gradients(loss, [future_1, future_2])
        velocity1 = velocity1.assign(gama * velocity1 + grad_1 * self.learning_rate_)
        velocity2 = velocity2.assign(gama * velocity2 + grad_2 * self.learning_rate_)
        new_future_1 = future_1.assign(w1 - gama * velocity1)
        new_future_2 = future_2.assign(w2 - gama * velocity2)
        new_w1 = w1.assign(w1 - velocity1)
        new_w2 = w2.assign(w2 - velocity2)
        return loss, tf.group(new_future_1, new_future_2, new_w1, new_w2)

    def _handcraft_optimizer(self, input, y_pred, y, w1, w2):
        loss = self._objective_loss(y_pred, y)
        regularization = tf.reduce_sum(self.rigid_ * (w1 * w1)) + tf.reduce_sum(self.rigid_ * (w2 * w2))
        loss_aug = tf.add(loss, 0.5 * regularization)

        if self.gradient_type_ == 'vanilia':
            return self._vanilia_gradient_descent(loss_aug, w1, w2)
        elif self.gradient_type_ == 'momentum':
            return self._momentum_gradient_descent(loss_aug, w1, w2) 
        elif self.gradient_type_ == 'nesterov':
            return self._nesterov_gradient_descent(input, y, w1, w2)
        else:
            raise ValueError('Not support gradient type')

    def _objective_loss(self, y_pred, y):
        pass

    def _relu(self, input, hidden_weight):
        if self.relu_ is None:
            return tf.maximum(tf.matmul(input, hidden_weight), 0)
        elif self.relu_ == 'leaky':
            x = tf.matmul(input, hidden_weight)
            hidden = tf.maximum(x, tf.scalar_mul(0.01, x))
            return hidden
        else:
            raise ValueError("Not supported relu type")

    def _is_convergent(self, loss):
        if loss > self.best_loss_value_ - self.tolerent_:
            self.no_improvement_num_ += 1
        else:
            self.no_improvement_num_ = 0

        if loss < self.best_loss_value_:
            self.best_loss_value_ = loss
        
        return self.no_improvement_num_ >= self.max_no_improvement_num_

    def _batch_normalization(self, input): 
        return input
        
        # mean = tf.reduce_mean(input, axis=0)
        # vars = tf.reduce_mean((input - mean) ** 2, axis=0)
        # eselon = 1e-8
        # return input - mean / tf.sqrt(vars + eselon)

    def _optimize(self, samples, labels):
        n_hidden = max(samples.shape[1] / 2, 8)
        n_hidden = int(floor(n_hidden))
        
        self.neuron_num_ = n_hidden
        self.n_features_ = samples.shape[1]
        self.n_labels_dim_ = labels.shape[1]
        w1, w2 = self._init_weights((samples.shape[1], n_hidden), (n_hidden, labels.shape[1]))

        x = tf.placeholder(tf.float32, shape=(self.batch_size_, samples.shape[1]), name="X")
        h = tf.Variable(w1)
        y = tf.Variable(w2)

        self.hidden_weight_ = h
        self.output_weight_ = y

        label = tf.placeholder(tf.float32, shape=(self.batch_size_, labels.shape[1]), name="Y")

        x = self._batch_normalization(x)
        hidden = self._relu(x, h)
        y_pred = tf.matmul(hidden, y)

        loss, optimize_action = None, None
        if self.gradient_type_ is not None:
            loss, optimize_action = self._handcraft_optimizer(x, y_pred, label, h, y)
        else:
            loss, optimize_action = self._predifined_optimizer(y_pred, label)
        
        self.sess_ = tf.Session()
        self.sess_.run(tf.global_variables_initializer())

        data = np.hstack((samples, labels)).astype(np.float32)
        n_batch = samples.shape[0] / self.batch_size_
        n_batch = int(floor(n_batch))
        for _ in range(self.max_iteration_):
            # batch stochastic gradient descent
            np.random.shuffle(data)
            batch_idx = 0
            for _ in range(n_batch):
                batch_data = data[self.batch_size_ * batch_idx: self.batch_size_ * (batch_idx + 1)]
                batch_samples = batch_data[:,:samples.shape[1]]
                batch_labels = batch_data[:,samples.shape[1]:]

                batch_samples -= np.mean(batch_samples, axis=0)
                # batch_samples /= np.std(batch_samples, axis=0)

                values = {
                    x: batch_samples,
                    label: batch_labels
                }
                loss_value, _ = self.sess_.run([loss, optimize_action], feed_dict=values)
                # print(loss_value)
                if self._is_convergent(loss_value):
                    # print(self.best_loss_value_)
                    return

        print("Failed to converge")

    def fit(self, samples, labels):
        n_samples = samples.shape[0]
        if n_samples < self.batch_size_:
            raise ValueError('invalid batch size which is bigger than samples size')

        self._optimize(samples, labels)

    def predict(self, X):
        samples = X - np.mean(X, axis=0)
        # samples /= np.std(samples, axis=0)

        input = tf.placeholder(tf.float32, X.shape)
        hidden = self._relu(input, self.hidden_weight_)
        y_pred = tf.matmul(hidden, self.output_weight_)

        value = {
            input: samples
        }
        y_pred_value = self.sess_.run(y_pred, feed_dict=value)
            
        return y_pred_value

if __name__ == '__main__':
    data = datasets.load_iris()
    X = data.data.astype(np.float32)
    y = data.target.reshape(X.shape[0], -1)
    print(X.shape[1])

    sk_mlp = neural_network.MLPClassifier(int(X.shape[0] / 3), max_iter=10000)
    sk_mlp.fit(X, y)
    sk_y_pred = sk_mlp.predict(X)
    sk_ac = metrics.accuracy_score(y, sk_y_pred)
    print(sk_ac)

    encoded_y = preprocessing.OneHotEncoder().fit_transform(y).toarray()
    # print(encoded_y)
    mlp = MLP(1e-4, None, relu='leaky', batch_size=90, tolerent=1e-5, rigid=0)
    mlp.fit(X, encoded_y)

    y_pred = mlp.predict(X)
    y_pred = np.argmax(y_pred, axis = 1)
    print(y_pred)
    print(y.reshape(1, -1))
    ac = metrics.accuracy_score(y, y_pred)
    print(ac)
    