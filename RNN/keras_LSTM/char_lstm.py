#!/usr/bin/env python

from tensorflow import keras
from tensorflow.python.lib.io import file_io

import numpy as np
import argparse
import os

class Keras_LSTM:
    def __init__(self, num_units, num_layers, batch_size, seq_len, vocab_size):
        self.model = keras.Sequential()
       
        # input
        self.model.add(keras.layers.Embedding(vocab_size, vocab_size, input_length=seq_len))
        
        # lstm
        for _ in range(num_layers):
            self.model.add(keras.layers.LSTM(num_units, dropout=0.5, return_sequences=True))

        # output
        self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(vocab_size)))
        self.model.add(keras.layers.Activation('softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        
    def fit(self, x, y):
        self.model.fit(x, y)

    def reset_states(self):
        self.model.reset_states()

    def predict(self, init_char):
        pred = self.model.predict(init_char)
        pred = np.squeeze(pred) 
        idx = np.argmax(pred[0])
        return idx

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
        x = np.zeros([batch_size, seq_len])
        y = np.zeros([batch_size, seq_len, vocab_len])
        
        offset = batch_idx * batch_size
        cur_idx = 0
        for i in range(batch_size):
            if cur_idx + offset + seq_len >= len(sample):
                raise StopIteration()

            x[i, :] = [char_to_idx[ch] for ch in sample[cur_idx + offset: cur_idx + offset + seq_len]]
            y[i, :, :] = [one_hot_encode(ch) for ch in sample[cur_idx + offset + 1: cur_idx + offset + seq_len + 1]]

            cur_idx += seq_len

        yield x, y
        
def main(batch_size,
         seq_len,
         num_epoches,
         num_units,
         num_layers,
         job_dir,
         input_file,
         output_file,
         output_len,
         check_point):
    with file_io.FileIO(input_file, 'r') as input:
        sample = list(input.read())

    output_file_name = os.path.join(job_dir, output_file)
    output = file_io.FileIO(output_file_name, 'w')

    char_to_idx, idx_to_char = build_dataset(sample)
    vocab_size = len(char_to_idx)

    model = Keras_LSTM(num_units, num_layers, batch_size, seq_len, vocab_size)

    for epoch in range(num_epoches):
        for inputs, targets in next_batch(sample, batch_size, seq_len, char_to_idx):
            model.fit(inputs, targets)
        model.reset_states()

        if epoch % check_point == 0 and epoch != 0:
            init_char = char_to_idx['b']
            pred_words = []
            output.write('[epoch %d] predicting...\n' % epoch)
            for _ in range(output_len):
                input = np.array([init_char])
                input = np.reshape(input, [-1, 1])
                pred = model.predict(input)
                pred_words.append(idx_to_char[pred])
                init_char = pred
            pred_str = "".join(pred_words)
            print("pred words: " + pred_str)
            output.write(pred_str)
            output.write("\n\n")

    init_char = char_to_idx['b']
    pred_words = []
    for _ in range(output_len):
        input = np.array([init_char])
        input = np.reshape(input, [-1, 1])
        pred = model.predict(input)
        pred_words.append(idx_to_char[pred])
        init_char = pred
    pred_str = "".join(pred_words)
    print("pred words: " + pred_str)
    output.write(pred_str)
    output.write("\n\n")
    output.close()


def parse_params(args_parser):
    args_parser.add_argument(
        '--job-dir',
        required=True
    )

    args_parser.add_argument(
        '--input-file',
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
        '--output-len',
        type=int,
        default=200
    )

    args_parser.add_argument(
        '--seq-len',
        type=int,
        required=True
    )

    return args_parser.parse_args()

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    paras = parse_params(args_parser)
    main(paras.batch_size, 
         paras.seq_len,
         paras.num_epoches,
         paras.num_units,
         paras.num_layers,
         paras.job_dir,
         paras.input_file,
         paras.output_file,
         paras.output_len,
         paras.check_point)