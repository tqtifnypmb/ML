import tensorflow as tf
import numpy as np
from math import floor

class MLP:
    def __init__(self, learning_rate, batch_size = 100, max_iteration = 10000, tolerent = 2.):
        self.learning_rate_ = learning_rate
        self.batch_size_ = batch_size
        self.max_iteration_ = max_iteration
        self.tolerent_ = tolerent

    def _init_weights(self, hidden_shape, output_shape):
        w1 = tf.random_normal(hidden_shape)
        w2 = tf.random_normal(output_shape)
        return w1, w2

    def _predifined_optimizer(self, y_pred, y):
        loss = tf.losses.mean_squared_error(y_pred, y)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_)
        optimize_action = optimizer.minimize(loss)

        return loss, optimize_action

    def _vanilia_gradient_descent(self, loss, w1, w2):
        grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])
        new_w1 = w1 - grad_w1 * self.learning_rate_
        new_w2 = w2 - grad_w2 * self.learning_rate_

        return loss, tf.group(new_w1, new_w2)

    def _handcraft_optimizer(self, y_pred, y, w1, w2):
        diff = y_pred - y
        loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))
        return self._vanilia_gradient_descent(loss, w1, w2)

    def _optimize(self, samples, labels, handcraft_optimizer=True):
        n_hidden = (samples.shape[1] + labels.shape[1]) / 3
        n_hidden = int(floor(n_hidden))

        w1, w2 = self._init_weights((samples.shape[1], n_hidden), (n_hidden, labels.shape[1]))

        x = tf.placeholder(tf.float32, shape=(self.batch_size_, samples.shape[1]), name="X")
        h = tf.Variable(w1)
        y = tf.Variable(w2)

        label = tf.placeholder(tf.float32, shape=(self.batch_size_, labels.shape[1]), name="Y")

        hidden = tf.maximum(tf.matmul(x, h), 0)
        y_pred = tf.matmul(hidden, y)

        loss, optimize_action = None, None
        if handcraft_optimizer:
            loss, optimize_action = self._handcraft_optimizer(y_pred, label, h, y)
        else:
            loss, optimize_action = self._predifined_optimizer(y_pred, label)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            data = np.hstack((samples, labels)).astype(np.float32)
            n_batch = samples.shape[0] / self.batch_size_
            n_batch = int(floor(n_batch))
            for _ in range(self.max_iteration_):
                last_loss_value = 0

                # batch stochastic gradient descent
                for _ in range(n_batch):
                    np.random.shuffle(data)
                    batch_data = data[:self.batch_size_]
                    batch_samples = batch_data[:,:samples.shape[1]]
                    batch_labels = batch_data[:,samples.shape[1]:]

                    values = {
                        x: batch_samples,
                        label: batch_labels
                    }
                    loss_value, _ = sess.run([loss, optimize_action], feed_dict=values)
                    print(loss_value)

                    if loss_value - last_loss_value <= self.tolerent_:
                        return

    def fit(self, samples, labels):
        n_samples = samples.shape[0]
        if n_samples < self.batch_size_:
            raise ValueError('invalid batch size which is bigger than samples size')

        self._optimize(samples, labels)

    def predict(self, X):
        pass

if __name__ == '__main__':
    N, D = 640, 1000
    X = np.random.randn(N, D)
    y = np.random.randn(N, D)

    mlp = MLP(1e-3)
    mlp.fit(X, y)