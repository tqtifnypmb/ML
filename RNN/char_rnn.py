import tensorflow as tf

class RNN: 
    def __init__(self, num_classes, hidden_size, num_layers, dropout_prob, seq_len, batch_size):
        cell, self.initial_state = self._build_lstm(hidden_size, num_layers, batch_size, dropout_prob)
        self.inputs, self.targets = self._build_inputs(batch_size, seq_len)
        outputs, self.final_state = tf.nn.dynamic_rnn(cell, tf.expand_dims(self.inputs, -1), initial_state=self.initial_state)
        logits = self._build_output(outputs, seq_len, batch_size, num_classes)

        self.loss = self._build_loss(logits, self.targets, num_classes)
     
    def _build_inputs(self, batch_size, seq_len):
        inputs = tf.placeholder(tf.float32, (batch_size, seq_len), 'inputs')
        targets = tf.placeholder(tf.float32, (batch_size, seq_len), 'targets')
        return inputs, targets

    def _build_lstm(self, hidden_size, num_layers, batch_size, dropout_prob):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=dropout_prob)
        cells = lstm_cell
        # cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range(num_layers)])
        initial_state = cells.zero_state(batch_size, tf.float32)
        return cells, initial_state

    def _build_loss(self, logits, targets, num_classes):
        encoded_targets = tf.one_hot(targets, num_classes)
        reshaped_targets = tf.reshape(encoded_targets, logits.shape)
        
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=reshaped_targets)
        return tf.reduce_mean(loss)

    def _build_output(self, output, seq_len, batch_size, num_classes):
        with tf.variable_scope('softmax'):
            w = tf.get_variable('w', (batch_size, seq_len))
            b = tf.get_variable('b', (num_classes,), initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(output, w) + b
        return logits

def batch_sample_generator(sample, batch_size):
    batch_num = int(len(sample) / batch_size)
    for batch_idx in range(batch_num):
        x = sample[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        y = x[1:] + [x[0]]
        if (batch_idx + 1) * batch_size < len(sample):
            y[-1] = sample[(batch_idx + 1) * batch_size + 1]
        else:
            y[-1] = 0
        yield x, y

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
    batch_size = 20
    model = RNN(len(index_dict), 4, 4, 0.5, 1, batch_size)
    
    with tf.Session() as sess:
        sess.initialize_all_variables()

        newest_state = sess.run(model.initial_state)
        optimizer = tf.train.AdamOptimizer()
        optimier_aciton = optimizer.minimize(model.loss)
        for batch_x, batch_y in batch_sample_generator(vocab_index, batch_size):
            feed = {
                model.inputs: batch_x.reshpae(model.inputs.shape),
                model.targets: batch_y.reshape(model.targets.shape),
                model.initial_state: newest_state,
            }

            batch_loss, _ = sess.run([model.loss, optimier_aciton], feed_dict=feed)
            newest_state = sess.run(model.final_state)

            print('batch loss %d' % batch_loss)