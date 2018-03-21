import numpy as np
from sklearn.neighbors import NearestNeighbors


def euclidean_distance(x, y):
    x_norm_sq = ((x ** 2).sum(axis=1))[:, np.newaxis]
    y_norm_sq = ((y ** 2).sum(axis=1))[np.newaxis, :]
    xy = x.dot(y.T)
    return (x_norm_sq - 2 * xy + y_norm_sq) ** (1/2)


def cosine_distance(x, y):
    x_norm = ((x ** 2).sum(axis=1))[:, np.newaxis] ** (1/2)
    y_norm = ((y ** 2).sum(axis=1))[np.newaxis, :] ** (1/2)
    xy = x.dot(y.T)
    return 1 - xy / (x_norm * y_norm)


class KNNClassifier:
    def __init__(self, k, strategy="brute", metric="euclidean", weights=True, test_block_size=256):
        self.eps = 10 ** (-5)
        self.k = k
        self.strategy = strategy
        self.model = None
        self.metric = None

        if strategy == 'brute':
            self.model = NearestNeighbors(algorithm='brute', metric=metric)
        elif strategy == 'kd_tree':
            self.model = NearestNeighbors(algorithm='kd_tree', metric=metric)
        elif strategy == 'ball_tree':
            self.model = NearestNeighbors(algorithm='ball_tree', metric=metric)

        if metric == 'euclidean':
            self.metric = euclidean_distance
        elif metric == 'cosine':
            self.metric = cosine_distance

        self.weights = weights
        self.test_block_size = test_block_size

        self.X = None
        self.y = None

    def fit(self, X, y):
        self.y = y
        if self.model is not None:
            self.model.fit(X, y)
            return

        self.X = X

    def find_kneighbors(self, X, return_distance):
        if self.model is not None:
            return self.model.kneighbors(X, self.k, return_distance)

        number_of_batches = \
            X.shape[0] // self.test_block_size + \
            int(X.shape[0] % self.test_block_size != 0)

        neighbors_indices = np.empty((0, self.k), np.int)
        neighbors_distances = np.empty((0, self.k), np.float)

        for batch in np.array_split(X, number_of_batches):
            distance_matrix = self.metric(batch, self.X)

            # [i, j] - j-th neighbor for i-th object
            # batch_indices = distance_matrix.argsort(axis=1)[:, :self.k]
            batch_indices = \
                distance_matrix.argpartition(np.arange(self.k), axis=1)[:, :self.k]
            neighbors_indices = np.vstack((neighbors_indices, batch_indices))

            if return_distance:
                # prepairing indices for distance matrix
                indices = (np.arange(batch.shape[0])[:, np.newaxis], batch_indices)
                batch_distances = distance_matrix[indices]
                neighbors_distances = np.vstack((neighbors_distances, batch_distances))

        if return_distance:
            return (neighbors_distances, neighbors_indices)
        else:
            return neighbors_indices

    def find_kneighbors_for_set(self, X_set, return_distance):
        nns = []

        for X in X_set:
            if self.model is not None:
                    dists, inds = self.model.kneighbors(X, self.k, True)
                    nns.append((dists, inds))
            else:
                nns.append(find_kneighbors(X, self.k, True))

        test_size = nns[0][0].shape[0]
        united_inds = np.empty((test_size, 0), np.int)
        united_dists = np.empty((test_size, 0), np.float)

        for nn in nns:
            united_inds = np.hstack((united_inds, nn[1]))
            united_dists = np.hstack((united_dists, nn[0]))

        sorted_inds = np.argpartition(united_dists, range(self.k), axis=1)[:, :self.k]
        indices = (np.arange(test_size)[:, np.newaxis], sorted_inds)

        if return_distance:
            return (united_dists[indices], united_inds[indices])
        else:
            return united_inds[indices]


    def predict(self, X):
        number_of_batches = \
            X.shape[0] // self.test_block_size + \
            int(X.shape[0] % self.test_block_size != 0)

        result = np.empty(0, np.int)

        for batch in np.array_split(X, number_of_batches):
            neighbors_distances, neighbors_indices = self.find_kneighbors(batch, True)
            neighbors_targets = self.y[neighbors_indices]

            classes = np.unique(neighbors_targets)

            # find scores of neighbors for each class
            classes_scores = np.empty((0, neighbors_targets.shape[0]), np.float)
            for c in classes:
                if self.weights:
                    weighted_scores = 1 / (neighbors_distances + self.eps) * (neighbors_targets == c)
                    scores = weighted_scores.sum(axis=1)
                else:
                    scores = (neighbors_targets == c).sum(axis=1)
                classes_scores = np.vstack((classes_scores, scores))
            classes_scores = classes_scores.T

            classes_indices = classes_scores.argmax(axis=1)
            result = np.r_[result, classes[classes_indices]]

        return result

    def predict_sets(self, X_sets):
        result = np.empty(0, np.int)

        neighbors_distances, neighbors_indices = self.find_kneighbors_for_set(X_sets, True)
        neighbors_targets = self.y[neighbors_indices]

        classes = np.unique(neighbors_targets)

        # find scores of neighbors for each class
        classes_scores = np.empty((0, neighbors_targets.shape[0]), np.float)
        for c in classes:
            if self.weights:
                weighted_scores = 1 / (neighbors_distances + self.eps) * (neighbors_targets == c)
                scores = weighted_scores.sum(axis=1)
            else:
                scores = (neighbors_targets == c).sum(axis=1)
            classes_scores = np.vstack((classes_scores, scores))
        classes_scores = classes_scores.T

        classes_indices = classes_scores.argmax(axis=1)
        result = np.r_[result, classes[classes_indices]]

        return result
