from sklearn import datasets
from sklearn import naive_bayes
import matplotlib.pyplot as plt

if __name__ == '__main__':
    iris = datasets.load_iris()
    gnb = naive_bayes.GaussianNB()
    y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
    print((y_pred != iris.target).sum())
    plt.scatter(iris.target, y_pred)
    plt.show()