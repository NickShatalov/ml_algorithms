import numpy as np
import time

from numpy import linalg as LA
from scipy.special import expit, logsumexp

import oracles


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    def __init__(self, loss_function, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        - 'multinomial_logistic' - многоклассовая логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_function = loss_function
        if self.loss_function == "binary_logistic":
            self.oracle = oracles.BinaryLogistic(**kwargs)
        elif self.loss_function == "multinomial_logistic":
            self.class_number = kwargs.get("class_number", None)
            self.oracle = oracles.MulticlassLogistic(**kwargs)

        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.w_0 = None
        self.w = None


    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        if self.loss_function == "multinomial_logistic" and self.class_number is None:
            self.class_number = np.max(np.unique(y)) + 1

        if trace:
            history = dict()
            history['time'] = []
            history['func'] = []
            history['accuracy'] = []

        if w_0 is not None:
            self.w_0 = w_0
        else:
            if self.loss_function == "binary_logistic":
                self.w_0 = np.zeros(X.shape[1])
            else:
                self.w_0 = np.zeros((self.class_number, X.shape[1]))

        self.w = self.w_0

        if trace:
            history['time'].append(0)
            prev_time = time.time()
            history['func'].append(self.get_objective(X, y))
            y_pred = self.predict(X)
            acc = (y_pred == y).sum() / len(y)
            history['accuracy'].append(acc)

        loss_prev = self.get_objective(X, y)
        for i in range(self.max_iter):
            grad = self.get_gradient(X, y)

            self.w = self.w - self.step_alpha / ((i + 1) ** self.step_beta) * grad
            loss = self.get_objective(X, y)

            if trace:
                history['time'].append(time.time() - prev_time)
                prev_time = time.time()
                history['func'].append(loss)
                y_pred = self.predict(X)
                acc = (y_pred == y).sum() / len(y)
                history['accuracy'].append(acc)

            if abs(loss - loss_prev) < self.tolerance:
                break
            loss_prev = loss

        if trace:
            return history


    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """
        if self.loss_function == "binary_logistic":
            res = np.sign(X.dot(self.w.T))
            res[res == 0] = 1
            return res
        elif self.loss_function == "multinomial_logistic":
            res = np.argmax(self.predict_proba(X), axis=1)
            return res


    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        if self.loss_function == "binary_logistic":
            p = expit(np.dot(X, self.w))
            return np.vstack((1 - p, p)).T
        elif self.loss_function == "multinomial_logistic":
            margins = X.dot(self.w.T).T
            log_P = margins - np.max(margins, axis=0) \
                - logsumexp(margins - np.max(margins, axis=0), axis=0)
            P = np.exp(np.clip(log_P, -10 ** 18, 10 ** 2.5))
            return P.T


    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        if self.w is not None:
            return self.oracle.func(X, y, self.w)
        else:
            raise Exception('Not fitted')


    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        if self.w is not None:
            return self.oracle.grad(X, y, self.w)
        else:
            raise Exception('Not fitted')


    def get_weights(self):
        """
        Получение значения весов функционала
        """
        if self.w is not None:
            return self.w
        else:
            raise Exception('Not fitted')


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function, batch_size, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, random_seed=None, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        - 'multinomial_logistic' - многоклассовая логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход


        max_iter - максимальное число итераций

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_function = loss_function
        if self.loss_function == "binary_logistic":
            self.l2_coef = kwargs.get('l2_coef', 0)
            self.oracle = oracles.BinaryLogistic(l2_coef=self.l2_coef)
        elif self.loss_function == "multinomial_logistic":
            self.class_number = kwargs.get('class_number', None)
            if self.class_number is None:
                raise Exception("class_number is not implemented")
            self.oracle = oracles.MulticlassLogistic(**kwargs)

        self.random_seed = kwargs.get('random_seed', None)
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed

        self.w_0 = None
        self.w = None


    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        l = X.shape[0]

        if self.loss_function == "multinomial_logistic" and self.class_number is None:
            self.class_number = np.max(np.unique(y)) + 1


        if trace:
            history = dict()
            history['epoch_num'] = []
            history['time'] = []
            prev_time = time.time()
            history['func'] = []
            history['weights_diff'] = []
            history['accuracy'] = []

        if w_0 is not None:
            self.w_0 = w_0
        else:
            if self.loss_function == "binary_logistic":
                self.w_0 = np.zeros(X.shape[1])
            else:
                self.w_0 = np.zeros((self.class_number, X.shape[1]))

        self.w = self.w_0

        batch_indices = np.random.permutation(l)[:self.batch_size]

        if trace:
            history['epoch_num'].append(0)
            history['time'].append(0)
            history['func'].append(self.get_objective(X, y))
            history['weights_diff'].append(0)
            w_prev = self.w
            y_pred = self.predict(X)
            acc = (y_pred == y).sum() / len(y)
            history['accuracy'].append(acc)

        epoch_prev = 0
        loss_prev = self.get_objective(X, y)
        for i in range(self.max_iter):
            grad = self.get_gradient(X[batch_indices], y[batch_indices])

            self.w = self.w - self.step_alpha / ((i + 1) ** self.step_beta) * grad

            epoch_num = (i + 1) * self.batch_size / l
            if abs(epoch_num - epoch_prev) >= log_freq:
                batch_indices = np.random.permutation(l)[:self.batch_size]
                loss = self.get_objective(X, y)
                if trace:
                    history['epoch_num'].append(epoch_num)
                    history['time'].append(time.time() - prev_time)
                    prev_time = time.time()
                    history['func'].append(loss)
                    history['weights_diff'].append(LA.norm(self.w - w_prev) ** 2)
                    w_prev = self.w
                    y_pred = self.predict(X)
                    acc = (y_pred == y).sum() / len(y)
                    history['accuracy'].append(acc)
                epoch_prev = epoch_num

                if abs(loss - loss_prev) < self.tolerance:
                    break
                loss_prev = loss

        if trace:
            return history
