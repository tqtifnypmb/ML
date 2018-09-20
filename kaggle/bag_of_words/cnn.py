import pandas as pd
import re
import nltk
import os
import numpy as np

from gensim.models import word2vec, KeyedVectors
from sklearn import model_selection
from tensorflow import keras
from bs4 import BeautifulSoup

def build_mode(vocab_size, num_words, num_features, num_conv, kernel_size, num_dense, num_units):
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

def review_to_sentence(raw_review, tokenizer, remove_stop_words = False):
    raw_review = raw_review.decode('utf-8')
    raw_sentences = tokenizer.tokenize(raw_review.strip())
    sentence = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            words = review_to_words(raw_sentence, remove_stop_words)
            sentence += words

    return sentence

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
    sentences = []
    for i in range(raw_reviews.size):
        print('encoding review: %.1f%%\r' % (float(i) / raw_reviews.size * 100))
        text = raw_reviews[i]
        sentence = review_to_sentence(text, tokenizer)
        sentences.append(sentence)
    return sentences

def train_and_save_word2vec(raw_reviews, num_features, tokenizer, output_file):
    sentences = encode_reviews_2(raw_reviews, tokenizer)
    model = word2vec.Word2Vec(sentences, size=num_features, min_count=1)
    model.init_sims(replace=True)
    model.wv.save_word2vec_format(output_file)
    return model, sentences

def vector_for_sentence(sent, num_features, model):
    vec = np.zeros([num_features])
    for word in sent:
        v = model[word]
        vec += v

    vec /= len(sent)
    return np.reshape(np.asarray(vec), [1, -1])

if __name__ == '__main__':
    train = read_data('data/labeledTrainData.tsv')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    wordModel = None
    num_features = 30
    word2vec_model_file = 'word2vec_model.model'
    sents = None
    if os.path.isfile(word2vec_model_file):
        sents = encode_reviews_2(train['review'], tokenizer)
        wordModel = KeyedVectors.load_word2vec_format(word2vec_model_file)
    else:
        wordModel, sents = train_and_save_word2vec(train['review'],
                                                   num_features,
                                                   tokenizer,
                                                   word2vec_model_file)

    idx_to_word = wordModel.wv.index2word

    batch_size = 32
    vocab_size = len(idx_to_word)
    num_conv = 1
    kernel_size = 3
    num_dense = 3
    num_units = 128
    num_words = 1
    model = build_mode(vocab_size, num_words, num_features, num_conv, kernel_size, num_dense, num_units)

    X = np.zeros([len(sents), 1, num_features])
    for i in range(len(sents)):
        sent = sents[i]
        vec = vector_for_sentence(sent, num_features, wordModel)
        X[i][0] = vec
    
    y = train['sentiment']
    model.fit(X, y, batch_size)

    # test
    test = read_data('data/testData.tsv')
    test_sents = encode_reviews_2(test['review'], tokenizer)
    wordModel.build_vocab(test_sents, update=True)
    test_x = np.zeros([len(test_sents), 1, num_features])
    for i in range(len(test_sents)):
        sent = test_sents[i]
        vec = vector_for_sentence(sent, num_features, wordModel)
        test_x[i][0] = vec

    pred = model.predict(test_x)
    pred = np.squeeze(pred)
    print(pred)
    output = pd.DataFrame( data={"id":test["id"], "sentiment":pred} )
    output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )