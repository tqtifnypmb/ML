from sklearn import datasets
from sklearn import naive_bayes
from sklearn import svm
import matplotlib.pyplot as plt

def naiveBayes():
    iris = datasets.load_iris()
    gnb = naive_bayes.GaussianNB()
    y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
    print((y_pred != iris.target).sum())
    plt.scatter(iris.target, y_pred)
    plt.show()

def svc():
    cls = svm.SVC()
    iris = datasets.load_iris()
    y_pred = cls.fit(iris.data, iris.target).predict(iris.data)
    print(cls.n_support_)
    print((y_pred != iris.target).sum())
    plt.scatter(iris.target, y_pred)
    plt.show()

if __name__ == '__main__':
    svc()