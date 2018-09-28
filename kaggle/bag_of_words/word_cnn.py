import pandas as pd
import numpy as np

from preprocessing import encode_data_by_word
from model import word_level_cnn

from StringIO import StringIO
from tensorflow import keras
from tensorflow.python.lib.io import file_io
from sklearn import model_selection
from gensim.models import KeyedVectors


def read_data(input):
    data = pd.read_csv(input,
                       header=0,
                       delimiter='\t',
                       quoting=3)
    return data

def main(train_file,
         test_file,
         output_file,
         num_words,
         batch_size,
         epoch):

    print('=========== Loading word2vec ===========')
    wordModel = KeyedVectors.load_word2vec_format('../google_300_model.bin', binary=True)

    train_input = StringIO(file_io.read_file_to_string(train_file))
    train = read_data(train_input)

    num_features = 300
    X = encode_data_by_word(train['review'], num_words, num_features, wordModel, True)
    y = keras.utils.to_categorical(train['sentiment'])

    model = word_level_cnn(num_words, num_features)
    model.fit(X, y, batch_size=batch_size, epochs=epoch, shuffle=False)

    test_input = StringIO(file_io.read_file_to_string(test_file))
    test = read_data(test_input)
    test_X = encode_data_by_word(test['review'], num_words, num_features, wordModel, True)

    print('========== do prediction ===============')
    pred = model.predict(test_X)
    pred = np.squeeze(pred)
    pred = np.argmax(pred, axis=1)

    output = pd.DataFrame(data={"id":test["id"], "sentiment":pred})
    result_file = file_io.FileIO(output_file, 'w')
    output.to_csv(result_file, index=False, quoting=3)
    result_file.close()