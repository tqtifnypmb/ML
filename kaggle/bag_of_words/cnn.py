import pandas as pd
import re
import nltk
import os

from sklearn import model_selection
from tensorflow import keras
from bs4 import BeautifulSoup

def build_mode(vocab_size, num_conv, kernel_size, num_dense, num_units):
    model = keras.Sequential()

    # input
    input = keras.layers.InputLayer(input_shape=(None, vocab_size))
    model.add(input)

    for _ in range(num_conv):
        # convoluntion
        conv = keras.layers.Convolution1D(1, kernel_size, activation='relu')
        model.add(conv)

        # dense
        pool = keras.layers.MaxPool1D()
        model.add(pool)

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

def text_codec():
    word_table = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
    idx_to_char = {i : ch for i, ch in enumerate(word_table)}
    char_to_idx = {ch : i for i, ch in enumerate(word_table)}

    return char_to_idx, idx_to_char

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review).get_text()

    letters_only = re.sub('[^a-zA-Z]', " ", review_text)

    words = letters_only.lower().split()

    stops = set(nltk.corpus.stopwords.words('english'))

    meaningful_words = [w for w in words if not w in stops]

    return " ".join(meaningful_words)

def read_data(file_name, output_file):
    if output_file is None:
        data = pd.read_csv(file_name,
                           header=0,
                           quoting=3)
        return data

    data = pd.read_csv(file_name,
                       header=0,
                       delimiter='\t',
                       quoting=3)

    reviews = data['review']
    clean_reviews = []
    for i in range(reviews.size):
        print('cleaning review: %.1f%%\r' % (float(i) / reviews.size * 100))
        text = review_to_words(reviews[i])
        clean_reviews.append(text)

    data['review'] = clean_reviews
    data.to_csv(output_file, index=False, quoting=3)
    return data

def encode_reviews(raw_reviews, char_to_idx):
    ret = []
    for i in range(raw_reviews.size):
        print('encoding review: %.1f%%\r' % (float(i) / raw_reviews.size * 100))
        text = raw_reviews[i]
        text = [char_to_idx[ch] for ch in text]
        ret.append(text)
    return ret

if __name__ == '__main__':
    char_to_idx, idx_to_char = text_codec()

    clean_data_file = 'cleanedLabeledTrainData.csv'
    train = None
    if os.path.isfile(clean_data_file):
        train = read_data(clean_data_file, None)
    else:
        train = read_data('data/labeledTrainData.tsv', 
                           clean_data_file)

    batch_size = 32
    vocab_size = len(char_to_idx)
    num_conv = 6
    kernel_size = 3
    num_dense = 3
    num_units = 128
    model = build_mode(vocab_size, num_conv, kernel_size, num_dense, num_units)

    # train_x, test_x, train_y, test_y = model_selection.train_test_split(train['review'], 
    #                                                                     train['sentiment'], 
    #                                                                     test_size=0.3)
    
    x = train['review']
    y = train['sentiment']
    model.fit(x, y, batch_size)
    # pred = model.predict(test_x, batch_size=batch_size)
    # print(pred)