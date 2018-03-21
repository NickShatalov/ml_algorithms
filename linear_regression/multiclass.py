import numpy as np


class MulticlassStrategy:
    def __init__(self, classifier, mode, **kwargs):
        """
        Инициализация мультиклассового классификатора

        classifier - базовый бинарный классификатор

        mode - способ решения многоклассовой задачи,
        либо 'one_vs_all', либо 'all_vs_all'

        **kwargs - параметры классификатор
        """
        self.classifier = classifier
        self.mode = mode
        self.kwargs = kwargs
        self.weights = None
        self.class_number = None

    def fit(self, X, y):
        """
        Обучение классификатора
        """
        self.class_number = self.kwargs.get("class_number", None)
        if self.class_number is None:
            self.class_number = np.max(np.unique(y)) + 1

        self.weights = np.empty((0, X.shape[1]), np.float)
        if self.mode == 'one_vs_all':
            for i in range(0, self.class_number):
                y_i = y.copy()
                y_i[y != i] = -1
                y_i[y == i] = 1
                self.classifier.fit(X, y_i)
                self.weights = np.vstack((self.weights, self.classifier.get_weights()))
        elif self.mode == 'all_vs_all':
            for s in range(0, self.class_number):
                for j in range(0, s):
                    X_sj = X[(y == s) + (y == j)]
                    y_sj = y[(y == s) + (y == j)]
                    y_sj[y_sj == s] = -1
                    y_sj[y_sj == j] = 1
                    self.classifier.fit(X_sj, y_sj)
                    self.weights = np.vstack((self.weights, self.classifier.get_weights()))


    def predict(self, X):
        """
        Выдача предсказаний классификатором
        """
        if self.mode == 'one_vs_all':
            return np.argmax(self.weights.dot(X.T), axis=0)
        elif self.mode == 'all_vs_all':
            ava = np.sign(self.weights.dot(X.T))
            i = 0
            for s in range(0, self.class_number):
                for j in range(0, s):
                    ava[i][ava[i] == 1] = j
                    ava[i][ava[i] == -1] = s
                    i += 1
            classes = np.empty((0, ava.shape[1]), np.float)
            for k in range(0, self.class_number):
                classes = np.vstack((classes, (ava == k).sum(axis=0)))
            return np.argmax(classes, axis=0)
