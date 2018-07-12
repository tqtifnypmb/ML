import tensorflow as tf
import numpy as np
from math import floor

class MLP:
    def __init__(self, learning_rate):
        self.learning_rate_ = learning_rate

    def fit(self, X, y):
        n_sample = X.shape[0]
        n_feature = X.shape[1]
        n_hidden = int(floor(n_sample / 3))

        input = tf.placeholder(tf.float32, shape=(n_sample, n_feature))
        output = tf.placeholder(tf.float32, shape=(n_sample, n_feature))

        # init weight randomly, which usually bring bad performance
        # w1 = tf.Variable(tf.random_normal((n_feature, n_hidden)))
        # w2 = tf.Variable(tf.random_normal((n_hidden, n_feature)))

        # hidden = tf.maximum(tf.matmul(input, w1), 0)
        # y_pred = tf.matmul(hidden, w2)

        # init weight using tensorflow
        init = tf.contrib.layers.xavier_initializer()
        hidden = tf.layers.dense(inputs=input, 
                                 units=n_hidden, 
                                 activation=tf.nn.relu,
                                 kernel_initializer=init)
        y_pred = tf.layers.dense(inputs=hidden, units=n_feature, kernel_initializer=init)

        # optimize manually
        # diff = y_pred - y
        # loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1))
        
        # grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])
        # new_w1 = w1.assign(w1 - self.learning_rate_ * grad_w1)
        # new_w2 = w2.assign(w2 - self.learning_rate_ * grad_w2)
        # update = tf.group(new_w1, new_w2)

        # optimize by tensorflow
        loss = tf.losses.mean_squared_error(y, y_pred)

        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_)
        update = optimizer.minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            values = {input: X,
                      output: y}
            
            for _ in range(5000):
                loss_val, _ = sess.run([loss, update], feed_dict=values)
                print(loss_val)

    def predict(self, X):
        pass

if __name__ == '__main__':
    N, D, H = 64, 1000, 100
    X = np.random.randn(N, D)
    y = np.random.randn(N, D)

    mlp = MLP(1e-5)
    mlp.fit(X, y)