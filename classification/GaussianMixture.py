from sklearn import mixture
from sklearn import datasets
from sklearn import model_selection
from scipy import linalg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import itertools

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.title(title)

if __name__ == '__main__':
    data,target = datasets.make_classification(n_samples=1000, n_features=2, n_redundant=0)
    train_x, test_x, train_y, test_y = model_selection.train_test_split(data, target, test_size=0.4)

    gm = mixture.GaussianMixture(n_components=2)
    gm.fit(train_x)
    y_pred = gm.predict(test_x)

    error = (y_pred != test_y).sum()
    print('num of error: {0}, ratio: {1}'.format(error, error / len(test_y)))

    plot_results(test_x, y_pred, gm.means_, gm.covariances_, 0, 'gm')
    #plt.scatter(train_x[:,0], train_x[:,1])
    plt.show()
    # _, axs = plt.subplots(2, 1)
    # axs[0].plot(test_y, label='train')
    # axs[0].legend(loc='lower right')
    # axs[1].plot(y_pred, label='predict')
    # axs[1].legend(loc='lower right')
    # plt.show()
    