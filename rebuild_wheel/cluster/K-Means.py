from sklearn import cluster
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters, n_iters = 10, max_iters = 300, tol = 1e-4):
        self.n_clusters_ = n_clusters
        self.n_iters_ = n_iters
        self.tol_ = tol
        self.max_iters_ = max_iters
    
    def _init_centers(self, X):
        # Choose n_clusters sample from X randomly
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)[:self.n_clusters_]
        return X[indices]

    def _cal_centers(self, X, labels):
        # cal new centers by averaging samples in same cluster

        assert(labels.shape[0] == X.shape[0])

        indices = {}
        for row_idx, label in enumerate(labels):
            if label not in indices.keys():
                indices[label] = [row_idx]
            else:
                indices[label].append(row_idx)

        new_centers = np.zeros((self.n_clusters_, X.shape[1]))
        for index in range(self.n_clusters_):
            indice = indices[index]
            cluster = X[indice]
            new_centers[index] = np.mean(cluster, axis=0)

        return new_centers

    def _cal_inertia(self, X, centers):
        # cal new labels
        assert(centers.shape[0] == self.n_clusters_)

        n_samples = X.shape[0]
        inertia = np.zeros(n_samples)
        labels = np.zeros(n_samples, dtype='int64')

        for row_idx, sample in enumerate(X):
            distances = self._cal_distances(sample, centers)
            label = np.argmin(distances)
            inertia[row_idx] = distances[label]
            labels[row_idx] = label
        
        return (labels, inertia)

    def _cal_distances(self, a, b):
        # Cal ||a - b||2
        return np.sqrt((a - b) ** 2).sum(axis=1)

    def _is_convergent(self, centers, new_centers):
        distances = self._cal_distances(centers, new_centers)
        return np.all(distances <= self.tol_)

    def _fit(self, X, centers):
        labels, inertias = None, None
        for _ in range(self.max_iters_):
            old_centers = centers.copy()
            labels, inertias = self._cal_inertia(X, centers)
            centers = self._cal_centers(X, labels)
            if self._is_convergent(old_centers, centers):
                break

        return centers, labels, inertias

    def fit(self, X):
        if X.shape[0] <= self.n_clusters_:
            raise ValueError('the number of samples is less than n_clusters')

        best_centers, best_labels, best_inertias = None, None, None
        for _ in range(self.n_iters_):
            centers = self._init_centers(X)

            if best_centers is None:
                best_centers, best_labels, best_inertias = self._fit(X, centers)
            else:
                new_centers, labels, inertias = self._fit(X, centers)
                if np.mean(inertias) < np.mean(best_inertias):
                    best_centers = new_centers
                    best_labels = labels
                    best_inertias = inertias
        self.centers_ = best_centers
        self.labels_ = best_labels

    def predict(self, X):
        if self.centers_ is None:
            raise ValueError('model training is not finished')

        labels, inertia = self._cal_inertia(X, self.centers_)
        return (labels, inertia)

if __name__ == '__main__':
    n_samples = 1500
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

    y_pred = cluster.KMeans(n_clusters=3, random_state=random_state).fit_predict(X)
    plt.subplot(211)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("sklearn")

    custom = KMeans(n_clusters=3)
    custom.fit(X)
    y_pred_2, _ = custom.predict(X)
    plt.subplot(212)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred_2)
    plt.title("custom")

    plt.show()