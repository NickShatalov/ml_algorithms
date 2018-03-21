import numpy as np
import scipy

from numpy import linalg as LA
from scipy.special import expit, logsumexp


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef=0):
        """
        Задание параметров оракула.

        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        return np.logaddexp(0, -y * X.dot(w)).mean() + \
            self.l2_coef / 2 * LA.norm(w) ** 2

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """

        l = X.shape[0]
        return -1 / l * X.T.dot(y * expit(-y * X.dot(w))) + self.l2_coef * w


class MulticlassLogistic(BaseSmoothOracle):
    """
    Оракул для задачи многоклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.

    w в этом случае двумерный numpy array размера (class_number, d),
    где class_number - количество классов в задаче, d - размерность задачи
    """

    def __init__(self, class_number=None, l2_coef=0):
        """
        Задание параметров оракула.

        class_number - количество классов в задаче

        l2_coef - коэффициент l2 регуляризации
        """
        self.class_number = class_number
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array (значения 1, 2, ... K)

        w - двумерный numpy array
        """
        if self.class_number is not None:
            class_number = self.class_number
        else:
            class_number = np.max(np.unique(y)) + 1


        l = X.shape[0]
        margins = X.dot(w.T).T

        log_P = margins[y, range(margins.shape[1])] - np.max(margins, axis=0) \
            - logsumexp(margins - np.max(margins, axis=0), axis=0)
        return -1 / l * log_P.sum() + self.l2_coef / 2 * LA.norm(w) ** 2


    def grad(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - двумерный numpy array
        """
        if self.class_number is not None:
            class_number = self.class_number
        else:
            class_number = np.max(np.unique(y)) + 1

        l = X.shape[0]
        margins = X.dot(w.T).T
        log_P = margins - np.max(margins, axis=0) \
            - logsumexp(margins - np.max(margins, axis=0), axis=0)
        P = np.exp(np.clip(log_P, -10 ** 18, 10 ** 10))

        c = np.zeros((X.shape[0], class_number))
        c[range(c.shape[0]), y] = 1

        return 1 / l * X.T.dot(P.T - c).T + self.l2_coef * w
