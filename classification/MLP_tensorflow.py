import tensorflow as tf
import numpy as np

from sklearn import datasets

if __name__ == '__main__':
    train, test = tf.keras.datasets.mnist.load_data()
    train_x, train_y = train
    print(train_x)