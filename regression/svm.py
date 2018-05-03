import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

if __name__ == '__main__':
    train_data = datasets.load_diabetes()
    rbf = svm.SVR()
    rbf.fit(train_data.data, train_data.target)
    y_pred = rbf.predict(train_data.data)

    _, axis = plt.subplots(2,1)
    axis[0].plot(train_data.target, label='train')
    axis[0].legend(loc='lower right')
    axis[1].plot(y_pred, label='predict')
    axis[1].legend(loc='lower right')
    plt.show()