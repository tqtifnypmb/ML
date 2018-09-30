import pandas as pd
import numpy as np
import nltk
import re

from bs4 import BeautifulSoup

def reviews_to_words(raw_reviews, skip_stop_words):
    stop_words = set(nltk.corpus.stopwords.words('english'))

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    num_reviews = raw_reviews.size
    letter_only_re = re.compile('[^a-zA-Z]')
    final_words_list = []
    print("======== tokenizing review into words ========")
    for i in range(num_reviews):
        print('encoding: %.1f%%\r' % (float(i) / raw_reviews.size * 100))

        review = raw_reviews[i]
        review_text = BeautifulSoup(review).get_text()
        sentences_list = tokenizer.tokenize(review_text.strip())

        words = []
        for j in range(len(sentences_list)):
            sents = sentences_list[j]
            sents = letter_only_re.sub(" ", sents)

            ws = sents.split(' ')
            ws = filter(lambda x: len(x) > 0, ws)
            if skip_stop_words:
                meaningful_words = [w for w in ws if not w in stop_words]
                words += meaningful_words
            else:
                words += ws
  
        final_words_list.append(words)

    return final_words_list

def reviews_to_chars(raw_reviews, char_to_idx, num_chars):
    chars_list = []

    print("======== encoding reviews into chars list ========")

    num_reviews = raw_reviews.size
    for i in range(num_reviews):
        print('encoding: %.1f%%\r' % (float(i) / num_reviews * 100))

        chars = []
        count = 0
        review = raw_reviews[i]
        review_text = BeautifulSoup(review).get_text()
        for c in review_text:
            c = c.lower()
            if c in char_to_idx:
                chars.append(char_to_idx[c])
                count += 1

                if count >= num_chars:
                    break

        count = len(chars)
        if count < num_chars:
            for _ in range(num_chars - count):
                chars.append(0)

        chars_list.append(chars)

    return chars_list

def words_list_to_vec(wordModel, words_list, num_words, num_features):
    vectors_list = []

    print("======== encoding words list into vectors ========")
    for i in range(len(words_list)):
        print('encoding: %.1f%%\r' % (float(i) / len(words_list) * 100))

        vectors = []
        words = words_list[i]
        for j in range(min(len(words), num_words)):
            word = words[j]
            if word not in wordModel:
                continue

            vec = wordModel[word]
            vectors.append(vec)

        num_vec = len(vectors)
        if num_vec < num_words:
            for _ in range(num_words - num_vec):
                vectors.append(np.zeros([num_features]))
        
        vectors_list.append(vectors)

    return vectors_list    

def words_list_to_chars(words_list, char_to_idx, num_chars):
    chars_list = []

    print("======== encoding words list into chars list ========")

    for i in range(len(words_list)):
        print('encoding: %.1f%%\r' % (float(i) / len(words_list) * 100))

        chars = []
        words = words_list[i]
        count = 0
        for word in words:
            for c in word:
                count += 1
                chars.append(char_to_idx[c.lower()])
                if count >= num_chars:
                    break

            if count >= num_chars:
                    break
        
        count = len(chars)
        if count < num_chars:
            for _ in range(num_chars - count):
                chars.append(0)

        chars_list.append(chars)

    return chars_list

def encode_data_by_word(raw_reviews, num_words, num_features, wordModel, skip_stop_words):
    nltk.download('punkt')

    words_list = reviews_to_words(raw_reviews, skip_stop_words)

    vectors_list = words_list_to_vec(wordModel, words_list, num_words, num_features)

    X = np.zeros([raw_reviews.size, num_words, num_features])
    
    for i in range(len(vectors_list)):
        vectors = vectors_list[i]

        for j in range(len(vectors)):
            vector = vectors[j]

            X[i][j] = vector

    return X

def encode_data_by_char(raw_reviews, num_chars, char_to_idx, skip_stop_words):
    # words_list = reviews_to_words(raw_reviews, skip_stop_words)

    chars_list = reviews_to_chars(raw_reviews, char_to_idx, num_chars) #words_list_to_chars(words_list, char_to_idx, num_chars)

    X = np.zeros([raw_reviews.size, num_chars])

    for i in range(len(chars_list)):
        chars = chars_list[i]

        for j in range(len(chars)):
            char = chars[j]
            X[i][j] = char

    return X