import tensorflow as tf
import numpy as np

class LSTM:
    def __init__(self, num_units, seq_len, batch_size, vocab_size):
        self.inputs = tf.placeholder(tf.float32, [None, seq_len, vocab_size], name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, vocab_size], name='targets')

        self.cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

        X = tf.unstack(self.inputs, seq_len, 1)
        self.hidden_state, _ = tf.nn.static_rnn(self.cell, X, dtype=tf.float32)
       
        self.weights = tf.Variable(tf.random_normal([num_units, vocab_size]))
        self.bias = tf.Variable(tf.random_normal([vocab_size]))

        Y = tf.matmul(self.hidden_state[-1], self.weights) + self.bias
        batch_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y, labels=self.targets)
        self.loss = tf.reduce_mean(batch_loss)

    def predict(self, sess, init_value, output_len, idx_to_char):
        print(self.cell.state_size)
        print(self.hidden_state[-1].shape)
        return
        state = self.hidden_state[-1]
        value = tf.placeholder(tf.float32, [None, 1], name='pred_value')
        cur_value = np.asarray([init_value]).reshape(-1, 1)

        pred = []
        for _ in range(output_len):
            outputs, state = self.cell(inputs=value, state=[1, state])
            print(outputs.shape)
            Y = tf.matmul(outputs[-1], self.weights) + self.bias
            
            feed_dict = {
                value: cur_value
            }
            py = sess.run([Y], feed_dict=feed_dict)
            idx = tf.argmax(py)
            cur_value = idx_to_char[idx]
            pred.append(cur_value)
            

def build_dataset(sample):
    unique_chars = list(set(sample))
    char_to_idx = { ch : i for i, ch in enumerate(unique_chars)}
    idx_to_char = { i : ch for i, ch in enumerate(unique_chars)}
    return char_to_idx, idx_to_char

def next_batch(sample, batch_size, seq_len, char_to_idx):
    vocab_len = len(char_to_idx)

    def one_hot_encode(ch):
        value = np.zeros([vocab_len])
        value[char_to_idx[ch]] = 1
        return value

    batch_len = batch_size * seq_len
    num_batch = int(len(sample) / batch_len)
    for i in range(num_batch):
        p = i * batch_len
        if p + batch_len + 1 >= len(sample):
            raise StopIteration()

        inputs = np.asarray([one_hot_encode(ch) for ch in sample[p: p + batch_len]])
        targets = np.asarray([one_hot_encode(ch) for ch in sample[p + seq_len + 1: p + batch_len + 2: seq_len]])

        yield np.reshape(inputs, [-1, seq_len, vocab_len]), np.reshape(targets, [-1, vocab_len])

if __name__ == '__main__':
    with open('sample_short.txt', 'r') as input:
        sample = list(input.read())

    char_to_idx, idx_to_char = build_dataset(sample)
    batch_size = 5
    seq_len = 10
    num_units = 128
    model = LSTM(num_units, seq_len, batch_size, len(char_to_idx))

    optimizer = tf.train.AdamOptimizer().minimize(model.loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for inputs, targets in next_batch(sample, batch_size, seq_len, char_to_idx):
            feed_dict = {
                model.inputs: inputs,
                model.targets: targets
            }

            _, loss = sess.run([optimizer, model.loss], feed_dict=feed_dict)

            print(loss)

        model.predict(sess, char_to_idx['a'], 10, idx_to_char)