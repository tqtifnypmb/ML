import pandas as pd
import re
import nltk
import os
import numpy as np

from StringIO import StringIO
from tensorflow import keras
from bs4 import BeautifulSoup
from tensorflow.python.lib.io import file_io

# char-level CNN
def build_mode(num_words, vocab_size, kernel_sizes, dense_shape, output_filters):
    model = keras.Sequential()

    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

    # embed
    embed = keras.layers.Embedding(vocab_size, vocab_size, input_length=num_words)
    model.add(embed)

    num_conv = len(kernel_sizes)
    for i in range(num_conv):
        # convoluntion
        kernel_size = kernel_sizes[i]
        conv = keras.layers.Conv1D(output_filters, 
                                   kernel_size, 
                                   activation='relu', 
                                   padding='valid',
                                   kernel_initializer=initializer)
        model.add(conv)

        # dense
        if i < 2:
            pool = keras.layers.MaxPool1D(pool_size=3, padding='valid')
            model.add(pool)

    pool = keras.layers.MaxPool1D(pool_size=3, padding='valid')
    model.add(pool)
    
    # flatten
    model.add(keras.layers.Flatten())

    # dropout
    dropout = keras.layers.Dropout(0.5)
    model.add(dropout)

    num_dense = len(dense_shape)
    for i in range(num_dense):
        # dense layers
        dense = keras.layers.Dense(dense_shape[i], activation='relu')
        model.add(dense)

        # dropout
        dropout = keras.layers.Dropout(0.5)
        model.add(dropout)

    # output
    output = keras.layers.Dense(2, activation='softmax')
    model.add(output)

    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

def review_to_words(raw_review, remove_stop_words):
    review_text = BeautifulSoup(raw_review).get_text()

    letters_only = re.sub('[^a-zA-Z]', " ", review_text)

    words = letters_only.lower().split()

    if remove_stop_words:
        stops = set(nltk.corpus.stopwords.words('english'))
        meaningful_words = [w for w in words if not w in stops]
        return ''.join(meaningful_words)
    else:
        return ''.join(words)

def preprocess_reviews(reviews, max_len, remove_stop_words, char_to_idx):
    reviews_str = []
    for i, review in enumerate(reviews):
        print('encoding review: %.1f%%\r' % (float(i) / reviews.size * 100))
        str = review_to_words(review, remove_stop_words)
        
        # ignore str longer than max_len
        if len(str) > max_len:
            str = str[:max_len]

        idx = [char_to_idx[ch] for ch in str]
        reviews_str.append(idx)

    # do padding, if needed
    for i in range(len(reviews_str)):
        str = reviews_str[i]
        if len(str) == max_len:
            continue

        for _ in range(max_len - len(str)):
            str += '0'
        
        reviews_str[i] = str

    return np.array(reviews_str)

def char_idx_map():
    vocab = '0abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    char_to_idx = {ch : idx for idx, ch in enumerate(vocab)}
    idx_to_char = {idx : ch for idx, ch in enumerate(vocab)}
    return char_to_idx, idx_to_char

def read_data(file_name):
    data = pd.read_csv(file_name,
                       header=0,
                       delimiter='\t',
                       quoting=3)
    return data

def main(train_file,
         test_file,
         output_file,
         nltk_dir,
         max_len,
         batch_size,
         epoch,
         output_filters):
    # nltk.download('stopwords', download_dir=nltk_dir)
    # nltk.data.path.append(nltk_dir)

    char_to_idx, _ = char_idx_map()

    train_input = StringIO(file_io.read_file_to_string(train_file))
    train = read_data(train_input)

    test_input = StringIO(file_io.read_file_to_string(test_file))
    test = read_data(test_input)

    X = preprocess_reviews(train['review'], max_len, False, char_to_idx)
    y = keras.utils.to_categorical(train['sentiment'])
    
    num_words = len(X[0])
    vocab_size = len(char_to_idx)
    kernel_sizes = [7, 7, 3, 3, 3, 3]
    dense_shape = [1024, 1024]
    cnn = build_mode(num_words, vocab_size, kernel_sizes, dense_shape, output_filters)
    cnn.fit(X, y, batch_size=batch_size, epochs=epoch, shuffle=True)

    test_X = preprocess_reviews(test['review'], max_len, False, char_to_idx)
    pred = cnn.predict(test_X)
    pred = np.squeeze(pred)
    print(pred)
    pred = np.argmax(pred, axis=1)
    
    output = pd.DataFrame(data={"id":test["id"], "sentiment":pred})
    output['sentiment'] = (output['sentiment'] > 0.5).astype('int')

    result_file = file_io.FileIO(output_file, 'w')
    output.to_csv(result_file, index=False, quoting=3)
    result_file.close()