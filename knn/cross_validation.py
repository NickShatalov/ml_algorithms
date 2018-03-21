import numpy as np

from nearest_neighbors import KNNClassifier


def kfold(n, n_folds):
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    result = []
    for i in range(n_folds):
        result.append((np.concatenate(folds[:i] + folds[i + 1:]), folds[i]))
    return result


def mod_kfold(folds, n_mods):
    n = len(folds[0][0]) + len(folds[0][1])
    new_folds = []
    for fold in folds:
        new_fold_0 = np.empty(0, np.int)
        new_fold_1 = np.empty(0, np.int)
        for i in range(n_mods):
            new_fold_0 = np.r_[new_fold_0, fold[0] + i * n]
            new_fold_1 = np.r_[new_fold_1, fold[1] + i * n]
        new_folds.append((new_fold_0, new_fold_1))
    return new_folds


def knn_cross_val_score(X, y, k_list, score, cv=None, **kwargs):
    def accuracy(y_true, y_pred):
        return (y_true == y_pred).sum() / len(y_true)

    if score == "accuracy":
        metric = accuracy

    if cv is None:
        cv = kfold(len(X), 3)

    test_block_size = kwargs.get("test_block_size", 256)
    weights = kwargs.get("weights", True)
    eps = 10 ** (-5)

    res_dict = dict()
    for k in k_list:
        res_dict[k] = []

    for fold in cv:
        X_train = X[fold[0]]
        X_validate = X[fold[1]]
        y_train = y[fold[0]]
        y_validate = y[fold[1]]

        knn = KNNClassifier(k_list[-1], **kwargs)
        knn.fit(X_train, y_train)
        neighbors_distances, neighbors_indices = knn.find_kneighbors(X_validate, True)

        for k in k_list:
            kn_distances = neighbors_distances[:, :k]
            kn_indices = neighbors_indices[:, :k]
            kn_targets = y_train[kn_indices]

            classes = np.unique(kn_targets)

            # find scores of neighbors for each class
            classes_scores = np.empty((0, kn_targets.shape[0]), np.float)
            for c in classes:
                if weights:
                    weighted_scores = 1 / (kn_distances + eps) * (kn_targets == c)
                    scores = weighted_scores.sum(axis=1)
                else:
                    scores = (kn_targets == c).sum(axis=1)
                classes_scores = np.vstack((classes_scores, scores))
            classes_scores = classes_scores.T

            classes_indices = classes_scores.argmax(axis=1)
            y_pred = classes[classes_indices]

            res_dict[k].append(metric(y_validate, y_pred))

    return res_dict

def knn_cross_val_score_mod(X, y, mods, k, score, cv=None, **kwargs):
    def accuracy(y_true, y_pred):
        return (y_true == y_pred).sum() / len(y_true)

    if score == "accuracy":
        metric = accuracy

    if cv is None:
        cv = kfold(len(X), 3)

    res = []

    for fold in cv:
        X_train = X[fold[0]]
        X_validate = X[fold[1]]

        X_validate_set = [X_validate]
        for mod in mods:
            X_validate_set.append(np.apply_along_axis(mod, axis=1, arr=X_validate))

        y_train = y[fold[0]]
        y_validate = y[fold[1]]

        knn = KNNClassifier(k, **kwargs)
        knn.fit(X_train, y_train)
        y_pred = knn.predict_sets(X_validate_set)
        res.append(accuracy(y_validate, y_pred))

    return np.array(res)
