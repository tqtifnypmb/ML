import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import datasets
from sklearn import svm

if __name__ == '__main__':
    data = datasets.load_diabetes()
    train_x, test_x, train_y, test_y = model_selection.train_test_split(data.data, data.target, test_size=0.3)
    rbf = svm.SVR()
    rbf.fit(train_x, train_y)
    y_pred = rbf.predict(test_x)

    _, axis = plt.subplots(2,1)
    axis[0].plot(test_y, label='train')
    axis[0].legend(loc='lower right')
    axis[1].plot(y_pred, label='predict')
    axis[1].legend(loc='lower right')
    plt.show()