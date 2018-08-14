import tensorflow as tf
import numpy as np

class RNN: 
    def __init__(self, num_classes, num_units, num_layers, seq_len, batch_size):
        self.inputs, self.targets = self._build_inputs(seq_len)
        self.cell = self._build_lstm_cell(num_units, num_layers)
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)

        x = tf.stack([self.inputs], 1)
        x = tf.reshape(self.inputs, [-1, seq_len])
        x = tf.split(x, seq_len, 1)
        y, state = tf.nn.static_rnn(self.cell, x, dtype=tf.float32)
        self.final_state = state

        self.weights = tf.random_normal([num_units, num_classes])
        self.bias = tf.Variable(tf.random_normal([num_classes]))
        output = tf.matmul(y[-1], self.weights) + self.bias

        encoded_targets = tf.one_hot(self.targets, num_classes, axis=1)
        self.loss = self._build_loss(output, encoded_targets)
        
    def _build_inputs(self, seq_len):
        x = tf.placeholder(tf.float32, [None, seq_len], name="inputs")
        y = tf.placeholder(tf.int32, [None, 1], name="targets")
        return x, y

    def _build_lstm_cell(self, unit_per_cell, num_of_layers):
        cell = tf.nn.rnn_cell.BasicLSTMCell(unit_per_cell)
        if num_of_layers == 1:
            return cell
        else:
            cells = tf.nn.rnn_cell.MultiRNNCell([cell for _ in range(num_of_layers)])
            return cells

    def _build_loss(self, y_pred, y_true):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
        cost = tf.reduce_mean(loss)
        return cost

def batch_sample_generator(sample, batch_size):
    batch_num = int(len(sample) / batch_size)
    for batch_idx in range(batch_num):
        x = sample[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        y = None
        if (batch_idx + 1) * batch_size < len(sample):
            y = [sample[(batch_idx + 1) * batch_size + 1]]
        else:
            y = [sample[0]]
        yield np.asarray(x), np.asarray(y)

def preprocess_sample(sample):
    duplicated = set()
    unique = [x for x in sample if not (x in duplicated or duplicated.add(x))]

    vocab_index = []
    for c in sample:
        idx = unique.index(c)
        vocab_index.append(idx)

    return vocab_index, unique

if __name__ == '__main__':
    with open('sample.txt', 'r') as input:
        vocab = list(input.read())

    vocab_index, index_dict = preprocess_sample(vocab)
    batch_size = 60
    num_classes = len(index_dict)
    seq_len = 20
    num_units = 20
    num_layers = 1
    model = RNN(num_classes, num_units, num_layers, seq_len, batch_size)
    
    with tf.Session() as sess:
        newest_state = sess.run(model.initial_state)
        optimizer = tf.train.AdamOptimizer().minimize(model.loss)
        sess.run(tf.global_variables_initializer())
        for batch_x, batch_y in batch_sample_generator(vocab_index, batch_size * seq_len):
            x = batch_x.reshape(-1, seq_len)
            y = batch_y.reshape(-1, 1)
            # x = np.split(x, seq_len, 1)
            feed = {
                model.inputs: x,
                model.targets: y,
                model.initial_state: newest_state,
            }

            batch_loss, _ = sess.run([model.loss, optimizer], feed_dict=feed)
            newest_state = sess.run(model.final_state)

            print('batch loss %d' % batch_loss)