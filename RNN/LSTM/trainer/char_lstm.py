#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import argparse
from tensorflow.python.lib.io import file_io

class LSTM:
    def __init__(self, num_units, num_layers, seq_len, batch_size, vocab_size):
        self.batch_size = batch_size
        self.inputs = tf.placeholder(tf.float32, [None, seq_len, vocab_size], name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, seq_len, vocab_size], name='targets')
        labels = tf.unstack(self.targets, seq_len, 1)

        if num_layers > 1:
            self.cell = tf.nn.rnn_cell.MultiRNNCell([self._build_cell(num_units) for _ in range(num_layers)])
        else:
            self.cell = self._build_cell(num_units)

        self.initial_states = self.cell.zero_state(batch_size, tf.float32)

        # initial_states = self._build_state_variables(self.cell, batch_size)

        hidden_state, self.final_state = tf.nn.dynamic_rnn(self.cell, self.inputs, initial_state=self.initial_states)
        hidden_state = tf.unstack(hidden_state, seq_len, 1)
       
        self.weights = tf.Variable(tf.random_normal([num_units, vocab_size]))
        self.bias = tf.Variable(tf.random_normal([vocab_size]))

        self.output = tf.matmul(hidden_state[-1], self.weights) + self.bias
        batch_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=labels)
        self.loss = tf.reduce_mean(batch_loss)

        # self.update_state = self._state_variables_update_op(initial_states, final_state)

        # zero_state = self._build_state_variables(self.cell, batch_size)
        # self.reset_state = self._state_variables_update_op(initial_states, zero_state)

    def _build_cell(self, num_units):
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
        return cell

    def _build_state_variables(self, cell, batch_size):
        zero_states = cell.zero_state(batch_size, tf.float32)
        if type(zero_states) is tf.nn.rnn_cell.LSTMStateTuple:
            c_var = tf.Variable(zero_states.c, trainable=False)
            h_var = tf.Variable(zero_states.h, trainable=False)
            return tf.nn.rnn_cell.LSTMStateTuple(c_var, h_var)
        else:
            variables = []
            for c, h in cell.zero_state(batch_size, tf.float32):
                c_var = tf.Variable(c, trainable=False)
                h_var = tf.Variable(h, trainable=False)
                lstm_tuple = tf.nn.rnn_cell.LSTMStateTuple(c_var, h_var)
                variables.append(lstm_tuple)
            return tuple(variables)

    def _state_variables_update_op(self, old_states, new_states):
        update_ops = []
        if type(new_states) is tf.nn.rnn_cell.LSTMStateTuple:
            update_c = tf.assign(old_states.c, new_states.c)
            update_h = tf.assign(old_states.h, new_states.h)
            update_ops.append(update_c)
            update_ops.append(update_h)
        else:
            for old_state, new_state in zip(old_states, new_states):
                update_c = tf.assign(old_state.c, new_state.c)
                update_h = tf.assign(old_state.h, new_state.h)
                update_ops.append(update_c)
                update_ops.append(update_h)
        return update_ops

    def predict(self, sess, init_value, output_len, seq_len, idx_to_char):
        cur_state = sess.run(self.cell.zero_state(self.batch_size, tf.float32))
      
        prev = init_value
        pred = []
        for i in range(output_len):
            print('generating character: %i ' % i)
        
            feed_dict = {
                self.inputs: prev,
                self.initial_states: cur_state,
            }

            py, state = sess.run([self.output, self.final_state], feed_dict=feed_dict)
            cur_state = state

            idx = np.argmax(py[-1])
            prev[0: len(prev) - 1] = prev[1: len(prev)]

            value = np.zeros_like(prev[-1][0])
            value[idx] = 1
            prev[-1] = value
           
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

    num_batch = int(len(sample) / (seq_len * batch_size))
    for batch_idx in range(num_batch):
        x = np.zeros([batch_size, seq_len, vocab_len])
        y = np.zeros([batch_size, seq_len, vocab_len])
        
        offset = batch_idx * batch_size
        cur_idx = 0
        for i in range(batch_size):
            if cur_idx + offset + seq_len >= len(sample):
                raise StopIteration()

            x[i, :, :] = [one_hot_encode(ch) for ch in sample[cur_idx + offset: cur_idx + offset + seq_len]]
            y[i, :, :] = [one_hot_encode(ch) for ch in sample[cur_idx + offset + 1: cur_idx + offset + seq_len + 1]]

            cur_idx += seq_len

        yield x, y

def main(input_file_name, 
         output_file_name,
         job_dir, 
         batch_size, 
         num_units, 
         num_layers, 
         learning_rate, 
         num_epoches,
         check_point,
         output_len,
         use_gpu):

    with file_io.FileIO(input_file_name, 'r') as input:
        sample = list(input.read())

    seq_len = 1
    output_file = os.path.join(job_dir, output_file_name)
    output = file_io.FileIO(output_file, 'w')

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

        print('training...')
        pred_inputs, _ = next(next_batch(sample, batch_size, seq_len, char_to_idx))
        for i in range(num_epoches):
            print('===============')
            print('epoch: %d' % i)
            print('===============')

            cur_state = sess.run(model.initial_states)
            for inputs, targets in next_batch(sample, batch_size, seq_len, char_to_idx):
                feed_dict = {
                    model.inputs: inputs,
                    model.targets: targets,
                    model.initial_states: cur_state
                }

                _, state, loss = sess.run([optimizer, model.final_state, model.loss], feed_dict=feed_dict)
                cur_state = state
                print(loss)

            if i != 0 and i % check_point == 0:
                output.write('[epoch %d] predicting...\n' % i)
                
                with tf.device(device_name):
                    pred = model.predict(sess, pred_inputs, output_len, seq_len, idx_to_char)
                output.write("".join(pred))
                output.write("\n\n")
                print("".join(pred))

            # reset state
            # sess.run([model.reset_state])

        print('predicting...\n')
        with tf.device(device_name):
            pred = model.predict(sess, pred_inputs, output_len, seq_len, idx_to_char)
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
         paras.num_units,
         paras.num_layers,
         paras.learning_rate,
         paras.num_epoches,
         paras.check_point,
         paras.output_len,
         paras.use_gpu)