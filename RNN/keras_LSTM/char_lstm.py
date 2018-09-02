#!/usr/bin/env python

from tensorflow import keras
from tensorflow.python.lib.io import file_io
import numpy as np
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
        pred = pred[:, -1, :]
        pred = np.squeeze(pred)
        
        idx = np.argmax(pred)
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

    cur_idx = 0
    for _ in range(len(sample)):
        x = np.zeros([batch_size, seq_len])
        y = np.zeros([batch_size, seq_len, vocab_len])
        for i in range(batch_size):
            if cur_idx + seq_len >= len(sample):
                cur_idx = 0

            x[i, :] = [char_to_idx[ch] for ch in sample[cur_idx: cur_idx + seq_len]]
            y[i, :, :] = [one_hot_encode(ch) for ch in sample[cur_idx + 1: cur_idx + seq_len + 1]]
            cur_idx += 1

        yield x, y
        
if __name__ == '__main__':
    with file_io.FileIO('../LSTM/data/sample_short.txt', 'r') as input:
        sample = list(input.read())

    output_file = os.path.join('.', 'output_file_name.txt')
    output = file_io.FileIO(output_file, 'w')

    char_to_idx, idx_to_char = build_dataset(sample)
    
    batch_size = 5
    num_epoches = 1
    seq_len =30
    num_units = 256
    num_layers = 2
    vocab_size = len(char_to_idx)

    model = Keras_LSTM(num_units, num_layers, batch_size, seq_len, vocab_size)

    for epoch in range(num_epoches):

        for inputs, targets in next_batch(sample, batch_size, seq_len, char_to_idx):
            model.fit(inputs, targets)

        model.reset_states()

    num_pred = 50
    pred_words = []

    pred_idx = 0
    for inputs, _ in next_batch(sample, 1, seq_len, char_to_idx):
        pred = model.predict(inputs)
        pred_words.append(idx_to_char[pred])   

        pred_idx += 1
        if (pred_idx > num_pred):
            break
    
    pred_str = "".join(pred_words)
    print("pred words: " + pred_str)