import tensorflow as tf
import numpy as np

class RNN: 
    def __init__(self, seq_len, num_classes, num_units, batch_size):
        self.inputs = tf.placeholder(tf.float32, [None, 1, seq_len], name="inputs")
        self.targets = tf.placeholder(tf.float32, [None, num_classes])
        self.cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

        x = tf.unstack(self.inputs, 1, axis=1)
        outputs, _ = tf.nn.static_rnn(self.cell, x, dtype=tf.float32)

        self.weight = tf.Variable(tf.random_normal([num_units, num_classes]))
        self.bias = tf.Variable(tf.random_normal([num_classes]))

        self.logits = tf.matmul(outputs[-1], self.weight) + self.bias

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, 
                                                                           labels=self.targets))
        
        pred = tf.nn.softmax(self.logits)
        pred = tf.reduce_mean(pred, 0)
        pred = tf.reshape(pred, (1,-1))
        self.prediction = tf.argmax(pred, 1)

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
    seq_len = 3
    num_units = 20
    num_iteration = 5
    model = RNN(seq_len, num_classes, num_units, batch_size)

    with tf.Session() as sess:
        optimizer = tf.train.AdamOptimizer().minimize(model.loss)
        sess.run(tf.global_variables_initializer())

        for i in range(num_iteration):
            for batch_x, batch_y in batch_sample_generator(vocab_index, batch_size * seq_len):
                x = batch_x.reshape(-1, 1, seq_len)
                y_one_hot = np.zeros([1, num_classes])
                y_one_hot[0][batch_y] = 1
                feed = {
                    model.inputs: x,
                    model.targets: y_one_hot,
                }

                batch_loss, _ = sess.run([model.loss, optimizer], feed_dict=feed)
                if i == num_iteration:
                    pred = sess.run([model.prediction], feed_dict=feed)
                    pred_idx = pred[0][0]
                    print('batch loss %d' % batch_loss)
                    print("{0} {1} {2} - {3}".format(index_dict[batch_x[0]],
                                                    index_dict[batch_x[1]],
                                                    index_dict[batch_x[2]],
                                                    index_dict[pred_idx]))