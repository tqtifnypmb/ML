import pandas as pd
import re
import nltk
import os
import numpy as np

from StringIO import StringIO
from tensorflow import keras
from bs4 import BeautifulSoup
from tensorflow.python.lib.io import file_io
from sklearn import model_selection

# char-level CNN
def build_mode(num_words, vocab_size):
    model = keras.Sequential()

    # embed
    embed = keras.layers.Embedding(vocab_size, vocab_size, input_length=num_words)
    model.add(embed)

    kernels = [11, 7, 5, 3, 3, 3]
    filters = [8, 16, 32, 32, 32, 32]
    has_norm = [True, True, False, False, False, False]
    has_pool = [True, True, False, False, False, False]
    dense_units = [1024, 1024]

    for i in range(len(kernels)):
        conv = keras.layers.Conv1D(filters[i], kernels[i], activation='relu', padding='same')
        model.add(conv)

        if has_norm[i]:
            norm = keras.layers.BatchNormalization()
            model.add(norm)
        
        if has_pool[i]:
            pool = keras.layers.MaxPool1D(pool_size=3, strides=2, padding='valid')
            model.add(pool)

    model.add(keras.layers.Flatten())

    for unit in dense_units:
        dense = keras.layers.Dense(unit, 'relu')
        model.add(dense)

        dropout = keras.layers.Dropout(0.5)
        model.add(dropout)

    output = keras.layers.Dense(2, activation='sigmoid')
    model.add(output)

    model.compile(optimizer='adam',
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    model.summary()
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
         epoch):
    # nltk.download('stopwords')
    # nltk.download('stopwords', download_dir=nltk_dir)
    # nltk.data.path.append(nltk_dir)

    char_to_idx, _ = char_idx_map()

    train_input = StringIO(file_io.read_file_to_string(train_file))
    train = read_data(train_input)

    test_input = StringIO(file_io.read_file_to_string(test_file))
    test = read_data(test_input)

    X = preprocess_reviews(train['review'], max_len, False, char_to_idx)
    y = keras.utils.to_categorical(train['sentiment'])
    
    train_X, valid_X, train_y, valid_y = model_selection.train_test_split(X, y, test_size=0.2)
    
    num_words = len(X[0])
    vocab_size = len(char_to_idx)
    cnn = build_mode(num_words, vocab_size)
    cnn.fit(train_X, train_y, validation_data=(valid_X, valid_y), batch_size=batch_size, epochs=epoch, shuffle=True)

    test_X = preprocess_reviews(test['review'], max_len, False, char_to_idx)
    pred = cnn.predict(test_X)
    pred = np.squeeze(pred)
    pred = np.argmax(pred, axis=1)
    
    output = pd.DataFrame(data={"id":test["id"], "sentiment":pred})
    output['sentiment'] = (output['sentiment'] > 0.5).astype('int')

    result_file = file_io.FileIO(output_file, 'w')
    output.to_csv(result_file, index=False, quoting=3)
    result_file.close()