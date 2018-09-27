import numpy as np

from sklearn import model_selection
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data

def build_model():
    model = keras.Sequential()

    kernels = [11, 7, 5, 3, 3, 3]
    filters = [8, 16, 32, 32, 32, 32]
    has_norm = [True, True, False, False, False, False]
    has_pool = [True, True, False, False, False, False]
    dense_units = [1024, 1024]

    input = keras.layers.InputLayer(input_shape=(28, 28, 1), name='input')
    model.add(input)

    for i in range(len(kernels)):
        conv = keras.layers.Conv2D(filters[i], kernels[i], activation='relu', padding='same')
        model.add(conv)

        if has_norm[i]:
            norm = keras.layers.BatchNormalization()
            model.add(norm)
        
        if has_pool[i]:
            pool = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')
            model.add(pool)

    model.add(keras.layers.Flatten())

    for unit in dense_units:
        dense = keras.layers.Dense(unit, 'relu')
        model.add(dense)

        dropout = keras.layers.Dropout(0.5)
        model.add(dropout)

    output = keras.layers.Dense(10, activation='sigmoid')
    model.add(output)

    model.compile(optimizer='adam',
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    model.summary()
    return model
    
if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_X, valid_X, train_y, valid_y = model_selection.train_test_split(mnist.train.images, 
                                                                        mnist.train.labels,
                                                                        test_size=0.3)
    test_X = np.reshape(mnist.test.images, (-1, 28, 28, 1))
    train_X = np.reshape(train_X, [-1, 28, 28, 1])
    valid_X = np.reshape(valid_X, (-1, 28, 28, 1))
    cnn = build_model()
    cnn.fit(train_X, train_y, batch_size=64, validation_data=(valid_X, valid_y))

    value = cnn.evaluate(test_X, mnist.test.labels, batch_size=64)
    print(value)