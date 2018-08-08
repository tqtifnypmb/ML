import tensorflow as tf
import numpy as np
from math import floor

class RNN: 
    def __init__(self, hidden_size, vocab_size, seq_len, batch_size):
        self.lstm_ = tf.contrib.rnn.BasicLSTMCell(seq_len)
        hidden_state = tf.zeros((batch_size, self.lstm_.state_size))
        current_state = tf.zeros((batch_size, self.lstm_.state_size))
        self.state_ = hidden_state, current_state
        self.x_ = tf.placeholder(tf.float32, shape=(batch_size, vocab_size))
        self.batch_size_ = batch_size

    def train(self, x):
        n_batch = x.shape[0] / self.batch_size_
        n_batch = int(floor(n_batch))
        for i in range(n_batch):
            batch_data = x[self.batch_size_ * i: self.batch_size_ * (i + 1)]
            output, self.state_ = self.lstm_(batch_data, self.state_)
            

    def predict(self, x):
        pass

if __name__ == '__main__':
    pass