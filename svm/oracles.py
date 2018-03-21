import numpy as np
import scipy


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


class BinaryHinge(BaseSmoothOracle):
    """
    Оракул для задачи двухклассового линейного SVM.

    Нулевая координата вектора w соответствует w_0.
    Считается, что в классификатор подаётся X с уже созданным единичным столбцом
    (так делается для того, чтобы не переписывать код из предыдущего задания).
    """

    def __init__(self, C=1.0):
        """
        Задание параметров оракула.

        C - коэффициент регуляризации в функционале SVM
        """
        self.C = C

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        hinge_loss = 1 - y * X.dot(w)
        hinge_loss[hinge_loss < 0] = 0.0
        # w[0] == w_0 - weight for constant feature
        return 1 / 2 * w[1:].dot(w[1:]) + self.C * hinge_loss.mean()

    def grad(self, X, y, w):
        """
        Вычислить субградиент функционала в точке w на выборке X с ответами y.
        Субгрдиент в точке 0 необходимо зафиксировать равным 0.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        l = X.shape[0]
        hinge_loss = 1 - y * X.dot(w)
        mask = (hinge_loss >= 0)
        res = -self.C / l * X[mask].T.dot(y[mask])
        # w[0] == w_0 - weight for constant feature
        res[1:] += w[1:]
        return res
