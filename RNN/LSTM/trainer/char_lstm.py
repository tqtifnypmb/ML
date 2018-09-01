#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import argparse
from tensorflow.python.lib.io import file_io

class LSTM:
    def __init__(self, num_units, num_layers, seq_len, batch_size, vocab_size):
        self.inputs = tf.placeholder(tf.float32, [None, seq_len, vocab_size], name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, vocab_size], name='targets')
        
        if num_layers > 1:
            self.cell = tf.nn.rnn_cell.MultiRNNCell([self._build_cell(num_units) for _ in range(num_layers)])
        else:
            self.cell = self._build_cell(num_units)

        initial_states = self._build_state_variables(self.cell, batch_size)

        hidden_state, final_state = tf.nn.dynamic_rnn(self.cell, self.inputs, initial_state=initial_states)
        hidden_state = tf.unstack(hidden_state, seq_len, 1)
       
        self.weights = tf.Variable(tf.random_normal([num_units, vocab_size]))
        self.bias = tf.Variable(tf.random_normal([vocab_size]))

        Y = tf.matmul(hidden_state[-1], self.weights) + self.bias
        batch_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y, labels=self.targets)
        self.loss = tf.reduce_mean(batch_loss)

        self.update_state = self._state_variables_update_op(initial_states, final_state)

        zero_state = self._build_state_variables(self.cell, batch_size)
        self.reset_state = self._state_variables_update_op(initial_states, zero_state)

    def _build_cell(self, num_units):
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
        return cell

    def _build_state_variables(self, cell, batch_size):
        variables = []
        for c, h in cell.zero_state(batch_size, tf.float32):
            c_var = tf.Variable(c, trainable=False)
            h_var = tf.Variable(h, trainable=False)
            lstm_tuple = tf.nn.rnn_cell.LSTMStateTuple(c_var, h_var)
            variables.append(lstm_tuple)
        return tuple(variables)

    def _state_variables_update_op(self, old_states, new_states):
        update_ops = []
        for old_state, new_state in zip(old_states, new_states):
            update_c = tf.assign(old_state.c, new_state.c)
            update_h = tf.assign(old_state.h, new_state.h)
            update_ops.append(update_c)
            update_ops.append(update_h)
        return update_ops

    def predict(self, sess, init_value, output_len, seq_len, idx_to_char):
        vocab_len = len(idx_to_char)
        cur_state = self.cell.zero_state(1, tf.float32) #self.final_state

        value = tf.placeholder(tf.float32, [None, vocab_len], name='pred_value')
      
        cur_value = np.zeros([vocab_len])
        cur_value[init_value] = 1
        
        pred = []
        for i in range(output_len):
            print('generating character: %i ' % i)

            outputs, state = self.cell(inputs=value, state=cur_state)
            cur_state = state
            
            Y = tf.matmul(outputs, self.weights) + self.bias
            
            feed_dict = {
                self.inputs: np.zeros([1, seq_len, vocab_len]),
                self.targets: np.zeros([1, vocab_len]),
                value: cur_value.reshape(-1, vocab_len)
            }

            py = sess.run([Y], feed_dict=feed_dict)
            idx = np.argmax(py)
     
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

def main(input_file_name, 
         output_file_name,
         job_dir, 
         batch_size, 
         seq_len, 
         num_units, 
         num_layers, 
         learning_rate, 
         num_epoches,
         check_point,
         output_len,
         use_gpu):

    with file_io.FileIO(input_file_name, 'r') as input:
        sample = list(input.read())

    output_file = os.path.join(job_dir, output_file_name)
    output = file_io.FileIO(output_file, 'w') #open(output_file, 'w')

    char_to_idx, idx_to_char = build_dataset(sample)

    # use gpu
    device_name = '/device:GPU:0'
    if use_gpu != 1:
        device_name = '/cpu:0'

    vocab_len = len(char_to_idx)
    with tf.device(device_name):
        model = LSTM(num_units, num_layers, seq_len, batch_size, vocab_len)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        dump_inputs = np.zeros([1, seq_len, vocab_len])
        dump_targets = np.zeros([1, vocab_len])

        print('training...')
        for i in range(num_epoches):
            print('===============')
            print('epoch: %d' % i)
            print('===============')
            for inputs, targets in next_batch(sample, batch_size, seq_len, char_to_idx):
                feed_dict = {
                    model.inputs: inputs,
                    model.targets: targets
                }

                _, _, loss = sess.run([optimizer, model.update_state, model.loss], feed_dict=feed_dict)
                print(loss)

            if i != 0 and i % check_point == 0:
                output.write('[epoch %d] predicting...\n' % i)
                init_char = 'b'
                with tf.device(device_name):
                    pred = model.predict(sess, char_to_idx[init_char], output_len, seq_len, idx_to_char)
                output.write("".join(pred))
                output.write("\n\n")
                print("".join(pred))

            # reset state
            sess.run([model.reset_state], feed_dict= {
                model.inputs: dump_inputs,
                model.targets: dump_targets
            })

        print('predicting...\n')
        init_char = 'b'
        with tf.device(device_name):
            pred = model.predict(sess, char_to_idx[init_char], output_len, seq_len, idx_to_char)
        output.write("".join(pred))
        output.write("\n\n")
        print("".join(pred))

    output.close()

def parse_params(args_parser):
    args_parser.add_argument(
        '--job-dir',
        required=True
    )

    args_parser.add_argument(
        '--train-file',
        required=True
    )

    args_parser.add_argument(
        '--output-file',
        required=True
    )

    args_parser.add_argument(
        '--batch-size',
        type=int,
        required=True
    )

    args_parser.add_argument(
        '--seq-len',
        type=int,
        required=True
    )

    args_parser.add_argument(
        '--num-units',
        type=int,
        required=True
    )

    args_parser.add_argument(
        '--num-layers',
        type=int,
        required=True,
    )

    args_parser.add_argument(
        '--num-epoches',
        type=int,
        required=True
    )

    args_parser.add_argument(
        '--check-point',
        type=int,
        required=True
    )

    args_parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
    )

    args_parser.add_argument(
        '--use-gpu',
        type=int,
        default=1
    )

    args_parser.add_argument(
        '--output-len',
        type=int,
        default=200
    )

    return args_parser.parse_args()

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    paras = parse_params(args_parser)
    main(paras.train_file, 
         paras.output_file,
         paras.job_dir, 
         paras.batch_size, 
         paras.seq_len,
         paras.num_units,
         paras.num_layers,
         paras.learning_rate,
         paras.num_epoches,
         paras.check_point,
         paras.output_len,
         paras.use_gpu)