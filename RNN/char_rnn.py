import tensorflow as tf
import numpy as np

class RNN: 
    def __init__(self, hidden_size, vocab_size, seq_len, batch_size):
        self.hidden_size_ = hidden_size
        self.vocab_size_ = vocab_size
        self.seq_len_ = seq_len
        self.batch_size_ = batch_size

        self.wxh_ = tf.Variable(tf.random_normal((batch_size, vocab_size)))
        self.whh_ = tf.Variable(tf.random_normal((hidden_size, hidden_size)))
        self.why_ = tf.Variable(tf.random_normal((hidden_size, vocab_size)))
        x = tf.placeholder(tf.float32, shape=(batch_size, vocab_size), name='X')

        self.h = np.zeros((hidden_size, hidden_size))
        xh = tf.matmul(x, self.wxh_, transpose_b=True)
        hh = tf.matmul(self.h, self.whh_, transpose_b=True)
        self.h = tf.nn.tanh(xh + hh)
        y = tf.matmul(self.h, self.why_, transpose_b=True)

    def train(self, x):
        pass

    def predict(self, x):
        pass

if __name__ == '__main__':
    pass