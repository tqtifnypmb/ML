import pandas as pd
import re
import nltk
import os

from gensim.models import word2vec, KeyedVectors
from sklearn import model_selection
from tensorflow import keras
from bs4 import BeautifulSoup

def build_mode(vocab_size, num_features, num_conv, kernel_size, num_dense, num_units):
    model = keras.Sequential()

    # input
    input = keras.layers.InputLayer(input_shape=[1, num_features])
    model.add(input)

    # embed = keras.layers.Embedding(vocab_size, vocab_size, input_length=1)
    # model.add(embed)

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

def review_to_words(raw_review, remove_stop_words = False):
    review_text = BeautifulSoup(raw_review).get_text()

    letters_only = re.sub('[^a-zA-Z]', " ", review_text)

    words = letters_only.lower().split()

    if remove_stop_words:
        stops = set(nltk.corpus.stopwords.words('english'))
        meaningful_words = [w for w in words if not w in stops]
        return meaningful_words
    else:
        return words

def review_to_sentences(raw_review, tokenizer, remove_stop_words = False):
    raw_review = raw_review.decode('utf-8')
    raw_sentences = tokenizer.tokenize(raw_review.strip())
    sentences = []
    total_words = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            words = review_to_words(raw_sentence, remove_stop_words)
            sentences.append(words)
            total_words += words

    return sentences, total_words

def read_data(file_name):
    data = pd.read_csv(file_name,
                       header=0,
                       delimiter='\t',
                       quoting=3)
    return data

def encode_reviews(raw_reviews, char_to_idx):
    ret = []
    for i in range(raw_reviews.size):
        print('encoding review: %.1f%%\r' % (float(i) / raw_reviews.size * 100))
        text = raw_reviews[i]
        text = [char_to_idx[ch] for ch in text]
        ret.append(text)
    return ret

def encode_reviews_2(raw_reviews, tokenizer):
    ret_sen = []
    ret_words = []
    for i in range(raw_reviews.size):
        print('encoding review: %.1f%%\r' % (float(i) / raw_reviews.size * 100))
        text = raw_reviews[i]
        sentences, words = review_to_sentences(text, tokenizer)
        ret_sen += sentences
        ret_words += words
    return ret_sen, ret_words

def train_and_save_word2vec(raw_reviews, num_features, output_file):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences, words = encode_reviews_2(train['review'], tokenizer)
    model = word2vec.Word2Vec(size=num_features)
    model.build_vocab(words, update=False)
    model.train(sentences, total_examples=len(sentences), epochs=1)
    model.init_sims(replace=True)
    model.wv.save_word2vec_format(output_file)
    return model

if __name__ == '__main__':
    train = read_data('data/labeledTrainData_truncated.tsv')

    wordModel = None
    num_features = 500
    word2vec_model_file = 'word2vec_model.model'
    if os.path.isfile(word2vec_model_file):
        wordModel = KeyedVectors.load_word2vec_format(word2vec_model_file)
    else:
        wordModel = train_and_save_word2vec(train['review'], 
                                            num_features, 
                                            word2vec_model_file)

    idx_to_word = wordModel.wv.index2word

    batch_size = 32
    vocab_size = len(idx_to_word)
    num_conv = 6
    kernel_size = 3
    num_dense = 3
    num_units = 128
    model = build_mode(vocab_size,num_features,  num_conv, kernel_size, num_dense, num_units)

    # train_x, test_x, train_y, test_y = model_selection.train_test_split(train['review'], 
    #                                                                     train['sentiment'], 
    #                                                                     test_size=0.3)
    
    x = wordModel.wv.syn0 #train['review']
    y = train['sentiment']
    model.fit(x, y, batch_size)
    # pred = model.predict(test_x, batch_size=batch_size)
    # print(pred)