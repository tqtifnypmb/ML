import tensorflow as tf
import numpy as np

class RNN: 
    def __init__(self, num_classes, hidden_size, num_layers, dropout_prob, vocab_size, seq_len, batch_size):
        cell, self.initial_state = self._build_lstm(hidden_size, num_layers, batch_size, dropout_prob)
        self.inputs, self.targets = self._build_inputs(batch_size, seq_len)

        outputs, self.final_state = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=self.initial_state)

        self.loss = self._build_loss(outputs, self.targets, num_classes)
     
    def _build_inputs(self, batch_size, seq_len):
        inputs = tf.placeholder(tf.float32, (batch_size, seq_len), 'inputs')
        targets = tf.placeholder(tf.float32, (batch_size, seq_len), 'targets')
        return inputs, targets

    def _build_lstm(self, hidden_size, num_layers, batch_size, dropout_prob):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=dropout_prob)
        cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range(num_layers)])
        initial_state = cells.zero_state(batch_size, tf.float32)
        return cells, initial_state

    def _build_loss(self, logits, targets, num_classes):
        encoded_targets = tf.one_hot(targets, num_classes)
        reshaped_targets = tf.reshape(encoded_targets, logits.shape)
        
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=reshaped_targets)
        return tf.reduce_mean(loss)

if __name__ == '__main__':
    pass