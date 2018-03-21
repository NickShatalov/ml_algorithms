import numpy as np
import math
import time

from numpy import linalg as LA

import oracles


class PEGASOSMethod:
    """
    Реализация метода Pegasos для решения задачи svm.
    """
    def __init__(self, step_lambda=1.0, batch_size=5, num_iter=10000):
        """
        step_lambda - величина шага, соответствует

        batch_size - размер батча

        num_iter - число итераций метода, предлагается делать константное
        число итераций
        """
        self.step_lambda = step_lambda
        self.batch_size = batch_size
        self.num_iter = num_iter

        self.w = None

    def get_objective(self, X, y):
        hinge_loss = 1 - y * X.dot(self.w)
        hinge_loss[hinge_loss < 0] = 0.0
        return self.step_lambda / 2 * self.w.dot(self.w) + hinge_loss.mean()

    def accuracy_score_(self, y_pred, y_test):
        return (y_pred == y_test).sum() / len(y_test)

    def fit(self, X, y, trace=False, log_freq = 0.1):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        d = X.shape[1]
        l = X.shape[0]

        self.w = np.zeros(d)

        if trace:
            history = dict()
            history['epoch_num'] = [0]
            history['time'] = [0]
            history['func'] = [self.get_objective(X, y)]
            history['weights_diff'] = [0]
            w_prev = self.w
            y_pred = self.predict(X)
            history['accuracy'] = [self.accuracy_score_(y_pred, y)]

        func_best = self.get_objective(X, y)
        epoch_prev = 0
        batch_indices = np.random.permutation(l)[:self.batch_size]
        start_time = time.time()
        for i in range(self.num_iter):
            alpha = 1 / ((i + 1) * self.step_lambda)
            mask = (X[batch_indices].dot(self.w) * y[batch_indices] < 1)
            X_ = X[batch_indices][mask]
            y_ = y[batch_indices][mask]
            self.w = (1 - alpha * self.step_lambda) * self.w + \
                alpha / self.batch_size * X_.T.dot(y_)
            tmp = self.step_lambda * self.w.dot(self.w)
            if tmp > 1:
                 self.w = 1 / math.sqrt(tmp) * self.w

            epoch_num = (i + 1) * self.batch_size / l
            if abs(epoch_num - epoch_prev) >= log_freq:
                batch_indices = np.random.permutation(l)[:self.batch_size]
                if trace:
                    history['epoch_num'].append(epoch_num)
                    history['time'].append(time.time() - start_time)
                    history['func'].append(self.get_objective(X, y))
                    history['weights_diff'].append(LA.norm(self.w - w_prev) ** 2)
                    w_prev = self.w
                    y_pred = self.predict(X)
                    history['accuracy'].append(self.accuracy_score_(y_pred, y))
                epoch_prev = epoch_num

        if trace:
            history['epoch_num'].append(epoch_num)
            history['time'].append(time.time() - start_time)
            history['func'].append(self.get_objective(X, y))
            history['weights_diff'].append(LA.norm(self.w - w_prev) ** 2)
            w_prev = self.w
            y_pred = self.predict(X)
            history['accuracy'].append(self.accuracy_score_(y_pred, y))
            return history

    def predict(self, X):
        """
        Получить предсказания по выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        """
        res = X.dot(self.w)
        res[res >= 0] = 1
        res[res < 0] = -1
        return res


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    def __init__(self, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """
        self.oracle = oracles.BinaryHinge(**kwargs)

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
        if trace:
            history = dict()
            history['time'] = []
            history['func'] = []
            history['accuracy'] = []

        if w_0 is not None:
            self.w_0 = w_0
        else:
            self.w_0 = np.zeros(X.shape[1])
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
        res = X.dot(self.w)
        res[res >= 0] = 1
        res[res < 0] = -1
        return res

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

    def __init__(self, batch_size=5, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, random_seed=None, **kwargs):
        """
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
        self.oracle = oracles.BinaryHinge(**kwargs)

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
            self.w_0 = np.zeros(X.shape[1])
        self.w = self.w_0



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
        batch_indices = np.random.permutation(l)[:self.batch_size]
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
