import pandas as pd
import re
import nltk
import os
import numpy as np

from gensim.models import word2vec, KeyedVectors
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from bs4 import BeautifulSoup

def build_mode(num_words, num_features, num_conv, kernel_size, num_dense, num_units):
    model = keras.Sequential()

    # input
    input = keras.layers.InputLayer(input_shape=[num_words, num_features], name='input')
    model.add(input)

    for _ in range(num_conv):
        # convoluntion
        conv = keras.layers.Conv1D(1, kernel_size, activation='relu', padding='same')
        model.add(conv)

        # dense
        pool = keras.layers.MaxPool1D(padding='same')
        model.add(pool)

        # dropout
        dropout = keras.layers.Dropout(0.25)
        model.add(dropout)

    # flatten
    model.add(keras.layers.Flatten())

    for _ in range(num_dense):
        # dense layers
        dense = keras.layers.Dense(num_units, activation='relu')
        model.add(dense)

    # output
    output = keras.layers.Dense(1, activation='sigmoid')
    model.add(output)

    model.compile(loss=keras.losses.binary_crossentropy, 
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

def read_data(file_name):
    data = pd.read_csv(file_name,
                       header=0,
                       delimiter='\t',
                       quoting=3)
    return data

data_set = {
    'train': 'data/labeledTrainData.tsv',
    'test': 'data/testData.tsv'
}

truncated_data_set = {
    'train': 'data/labeledTrainData_truncated.tsv',
    'test': 'data/testData_truncated.tsv'
}

if __name__ == '__main__':    
    data = data_set
    train = read_data(data['train'])
    test = read_data(data['test'])