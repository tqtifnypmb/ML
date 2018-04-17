import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

if __name__ == '__main__':
    x = 1.0 / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
    y = np.ones(10)

    n_alphas = 200
    alphas = np.logspace(-10, -2, n_alphas)

    coefs = []
    for a in alphas:
        ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
        ridge.fit(x, y)
        coefs.append(ridge.coef_)

    #plot
    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()