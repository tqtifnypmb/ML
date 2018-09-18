import numpy as np
import pandas as pd
import re
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review).get_text()

    letters_only = re.sub('[^a-zA-Z]', " ", review_text)

    words = letters_only.lower().split()

    stops = set(nltk.corpus.stopwords.words('english'))

    meaningful_words = [w for w in words if not w in stops]

    return " ".join(meaningful_words)

def vectorizer(raw_reviews):
    clean_train_reviews = []
    for i in range(raw_reviews.size):
        clean_words = review_to_words(raw_reviews[i])
        clean_train_reviews.append(clean_words)

    vectorizer = CountVectorizer(analyzer='word',
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)
    features = vectorizer.fit_transform(clean_train_reviews)
    features = features.toarray()
    return features

if __name__ == '__main__':
    train = pd.read_csv('data/labeledTrainData.tsv',
                        header=0,
                        delimiter='\t',
                        quoting=3)

    train_data_features = vectorizer(train['review'])
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, train['sentiment'])

    test = pd.read_csv('data/testData.tsv',
                        header=0,
                        delimiter='\t',
                        quoting=3)
    test_data_features = vectorizer(test['review'])
    result = forest.predict(test_data_features)

    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )