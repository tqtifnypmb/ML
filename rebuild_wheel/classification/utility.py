import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection

def group_features(X, Y):
    """
    Split features X into Xs which are labeled
    as same class Y

    return {class: [features row idx]}
    """

    if X.shape[0] != Y.shape[0]:
        raise ValueError('inconsistent X and Y')

    class_idx = {}
    idx = 0
    for kls in Y:
        if kls in class_idx.keys():
            class_idx[kls].append(idx)
        else:
            class_idx[kls] = [idx]
        idx += 1

    return class_idx

def test_model(nb, custom_nb):
    iris = datasets.load_iris()
    train_x, test_x, train_y, test_y = model_selection.train_test_split(iris.data, iris.target, test_size=0.4)
    _, axs = plt.subplots(3, 1)

    axs[0].plot(test_y, label='Test data')
    axs[0].legend(loc='lower right')

    nb.fit(train_x, train_y)
    y_pred = nb.predict(test_x)
    axs[1].plot(y_pred, label='SciKit model')
    axs[1].legend(loc='lower right')

    custom_nb.fit(train_x, train_y)
    y_pred_2 = custom_nb.predict(test_x)
    axs[2].plot(y_pred_2, label='Custom model')
    axs[2].legend(loc='lower right')

    plt.show()