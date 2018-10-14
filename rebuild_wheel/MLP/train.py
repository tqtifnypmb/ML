import numpy as np

from models import MLP
from sklearn import datasets
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':
    iris = datasets.load_iris()
    train_x, test_x, train_y, test_y = model_selection.train_test_split(iris.data, iris.target, test_size=0.4)

    mlp = MLP(train_x.shape, [50,50,3])

    num_iteration = 100
    for _ in range(num_iteration):
        mlp.fit(train_x, train_y)

    pred_y = mlp.predict(test_x)

    pred = np.argmax(pred_y, axis=1)
    print(pred)
    acc = metrics.mean_squared_error(test_y, pred)
    print(acc)

    # sMLP = MLPClassifier([50,50])
    # sMLP.fit(train_x, train_y)

    # pred = sMLP.predict(test_x)
    # print(pred)
    # acc = metrics.mean_squared_error(test_y, pred)
    # print(acc)