import tensorflow as tf
import numpy as np

from tensorflow import keras
from math import floor
from sklearn import datasets
from sklearn import preprocessing

class MLP:
    def __init__(self, learning_rate = 1e-3):
        self.learning_rate_ = learning_rate
        self.mlp_ = None

    def _create_model(self, n_features, n_hidden, n_classes):
        model = keras.Sequential()

        activation_layer = keras.layers.Dense(n_hidden, input_shape=(n_features,), activation='relu')
        model.add(activation_layer)

        output_layer = keras.layers.Dense(n_classes, activation='softmax')
        model.add(output_layer)

        optimizer = tf.train.AdamOptimizer(self.learning_rate_)
        model.compile(optimizer, 
                      loss=tf.losses.hinge_loss,
                      metrics=['accuracy'])

        self.mlp_ = model

    def fit(self, X, y):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_hidden = max(n_samples / 2, 4)
        n_hidden = int(floor(n_hidden))
        n_classes = y.shape[1]
        if self.mlp_ is None:
            self._create_model(n_features, n_hidden, n_classes)
        
        self.mlp_.fit(X, y, batch_size=80, epochs=200)

    def predict(self, X):
        return self.mlp_.predict(X)

    def evaluate(self, X, y):
        return self.mlp_.evaluate(X, y)

if __name__ == '__main__':
    data = datasets.load_iris()
    X = data.data
    y = data.target.reshape(-1, 1)
    y = preprocessing.OneHotEncoder().fit_transform(y).toarray()

    mlp = MLP(1e-2)
    mlp.fit(X, y)
    loss, acc = mlp.evaluate(X, y)
    print(acc)
    print(loss)