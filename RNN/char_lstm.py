import tensorflow as tf
import numpy as np

class LSTM:
    def __init__(self, num_units, seq_len, batch_size, vocab_size):
        self.inputs = tf.placeholder(tf.float32, [None, seq_len, vocab_size], name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, vocab_size], name='targets')

        self.cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

        X = tf.unstack(self.inputs, seq_len, 1)
        hidden_state, _ = tf.nn.static_rnn(self.cell, X, dtype=tf.float32)
       
        self.weights = tf.Variable(tf.random_normal([num_units, vocab_size]))
        self.bias = tf.Variable(tf.random_normal([vocab_size]))

        Y = tf.matmul(hidden_state[-1], self.weights) + self.bias
        batch_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y, labels=self.targets)
        self.loss = tf.reduce_mean(batch_loss)

    def predict(self, sess, init_value, output_len, seq_len, idx_to_char):
        vocab_len = len(idx_to_char)
        h_state = np.zeros([1, self.cell.state_size.h], dtype=np.float32)
        c_state = np.zeros([1, self.cell.state_size.c], dtype=np.float32)

        value = tf.placeholder(tf.float32, [None, vocab_len], name='pred_value')
      
        cur_value = np.zeros([vocab_len])
        cur_value[init_value] = 1
        
        pred = []
        for _ in range(output_len):
            outputs, state = self.cell(inputs=value, state=(c_state, h_state))
            c_state, h_state = state
            
            Y = tf.matmul(outputs, self.weights) + self.bias
            
            feed_dict = {
                self.inputs: np.zeros([1, seq_len, vocab_len]),
                self.targets: np.zeros([1, vocab_len]),
                value: cur_value.reshape(-1, vocab_len)
            }

            py = sess.run([Y], feed_dict=feed_dict)
            idx = np.argmax(py[0])
     
            cur_value = np.zeros_like(cur_value)
            cur_value[idx] = 1

            char = idx_to_char[idx]
            pred.append(char)

        return pred
            
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
    with open('sample.txt', 'r') as input:
        sample = list(input.read())

    char_to_idx, idx_to_char = build_dataset(sample)
    batch_size = 50
    seq_len = 200
    num_units = 128
    model = LSTM(num_units, seq_len, batch_size, len(char_to_idx))
    learning_rate = 1e-2
    num_epoches = 10

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(num_epoches):
            for inputs, targets in next_batch(sample, batch_size, seq_len, char_to_idx):
                feed_dict = {
                    model.inputs: inputs,
                    model.targets: targets
                }

                _, loss = sess.run([optimizer, model.loss], feed_dict=feed_dict)
                print(loss)

        init_char = 'a'
        pred = model.predict(sess, char_to_idx[init_char], 200, seq_len, idx_to_char)
        print(init_char.join(pred))