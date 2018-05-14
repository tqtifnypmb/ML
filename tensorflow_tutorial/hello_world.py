import numpy as np
import tensorflow as tf
import pandas as pd
import os

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

def create_train_data():
    iris = pd.read_csv('./iris_training.csv', names=CSV_COLUMN_NAMES, header=0)
    features, labels = iris, iris.pop('Species')
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.shuffle(1000).repeat().batch(32)

def loss(model, x, y):
    y_pred = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_pred)

def grad(model, features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss(model, features, labels)
        return tape.gradient(loss_value, model.variable)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    tf.enable_eager_execution()

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(3)
    ])

    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    dataset = create_train_data()
    epochs = 201

    for _ in range(epochs):
        for x, y in dataset:
            grads = grad(model, x, y)
            opt.apply_gradients(zip(grads, model.variables),global_step=tf.train.get_or_create_global_step())