import pandas as pd
import numpy as np

from preprocessing import encode_data_by_char
from model import char_level_cnn

from tensorflow import keras
from StringIO import StringIO
from tensorflow.python.lib.io import file_io
from sklearn import model_selection

def char_idx_map():
    vocab = ' abcdefghijklmnopqrstuvwxyz0123456789,./?\'(){}[]<>=-+_~:'
    char_to_idx = {ch : idx for idx, ch in enumerate(vocab)}
    return char_to_idx

def read_data(file_name):
    data = pd.read_csv(file_name,
                       header=0,
                       delimiter='\t',
                       quoting=3)
    return data

def main(train_file,
         test_file,
         output_file,
         num_chars,
         batch_size,
         epoch):

    char_to_idx = char_idx_map()
    vocab_size = len(char_to_idx)

    cnn = char_level_cnn(num_chars, vocab_size)

    train_input = StringIO(file_io.read_file_to_string(train_file))
    train = read_data(train_input)

    test_input = StringIO(file_io.read_file_to_string(test_file))
    test = read_data(test_input)

    X = encode_data_by_char(train['review'], num_chars, char_to_idx, False, one_hot=True)
    X = np.reshape(X, [-1, num_chars, 1, vocab_size])
    y = keras.utils.to_categorical(train['sentiment'])
    
    cnn.fit(X, y, batch_size=batch_size, epochs=epoch, shuffle=True)
    
    test_X = encode_data_by_char(test['review'], num_chars, char_to_idx, False, one_hot=True)
    test_X = np.reshape(test_X, [-1, num_chars, 1, vocab_size])

    print('========== do prediction ===============')

    pred = cnn.predict(test_X)
    pred = np.squeeze(pred)
    pred = np.argmax(pred, axis=1)
    
    output = pd.DataFrame(data={"id":test["id"], "sentiment":pred})
    result_file = file_io.FileIO(output_file, 'w')
    output.to_csv(result_file, index=False, quoting=3)
    result_file.close()