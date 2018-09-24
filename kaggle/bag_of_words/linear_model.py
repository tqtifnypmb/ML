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

def vector_for_sentence(sent, num_features, model):
    vec = np.zeros([num_features])
    count = 0
    for word in sent:
        if word in model:
            v = model[word]
            vec += v
            count += 1
    if count > 0:
        vec /= count
    return np.reshape(np.asarray(vec), [1, -1])

def vector_for_sentences(sents, num_features, wordModel):
    X = np.zeros([len(sents), num_features])
    for i in range(len(sents)):
        sent = sents[i]
        vec = vector_for_sentence(sent, num_features, wordModel)
        X[i] = vec
    return X

def train_and_save_word2vec(raw_reviews, num_features, tokenizer, remove_stop_words=False):
    sentences = encode_reviews(raw_reviews, tokenizer, remove_stop_words)
    model = word2vec.Word2Vec(sentences, size=num_features, min_count=1)
    model.init_sims(replace=True)
    return model, sentences

def review_to_words(raw_review, remove_stop_words):
    review_text = BeautifulSoup(raw_review).get_text()

    letters_only = re.sub('[^a-zA-Z]', " ", review_text)

    words = letters_only.lower().split()

    if remove_stop_words:
        stops = set(nltk.corpus.stopwords.words('english'))
        meaningful_words = [w for w in words if not w in stops]
        return meaningful_words
    else:
        return words

def review_to_sentence(raw_review, tokenizer, remove_stop_words):
    # raw_review = raw_review.decode('utf-8')
    raw_sentences = tokenizer.tokenize(raw_review.strip())
    sentence = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            words = review_to_words(raw_sentence, remove_stop_words)
            sentence += words

    return sentence

def encode_reviews(raw_reviews, tokenizer, remove_stop_words=False):
    sentences = []
    for i in range(raw_reviews.size):
        print('encoding review: %.1f%%\r' % (float(i) / raw_reviews.size * 100))
        text = raw_reviews[i]
        sentence = review_to_sentence(text, tokenizer, remove_stop_words)
        sentences.append(sentence)
    return sentences

def do_predict(train, test, num_features, forest, remove_stop_words, pretrained_word2vec=True):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    wordModel = None
    train_sents = None
    if pretrained_word2vec:
        wordModel = KeyedVectors.load_word2vec_format('./google_300_model.bin', binary=True)
        train_sents = encode_reviews(train['review'], tokenizer, remove_stop_words)
    else:
        wordModel, train_sents = train_and_save_word2vec(train['review'], num_features, tokenizer)
    
    X = vector_for_sentences(train_sents, num_features, wordModel)    
    y = train['sentiment']
    forest.fit(X, y)

    test_sents = encode_reviews(test['review'], tokenizer, remove_stop_words)
    X = vector_for_sentences(test_sents, num_features, wordModel)
    pred = forest.predict(X)
    
    return pred

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
    data = truncated_data_set
    train = read_data(data['train'])
    test = read_data(data['test'])
    num_features = 300

    # logistics = LogisticRegression()
    # mlp = MLPClassifier(hidden_layer_sizes=(300, 300, 300))
    forest = RandomForestClassifier(n_estimators=200)
    pred = do_predict(train, test, num_features, forest, remove_stop_words=False, pretrained_word2vec=False)

    output = pd.DataFrame(data={"id":test["id"], "sentiment":pred})    
    output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3)