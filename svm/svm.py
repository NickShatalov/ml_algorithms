import numpy as np
from cvxopt import solvers, matrix
from scipy.spatial import distance_matrix


class SVMSolver:
    """
    Класс с реализацией SVM через метод внутренней точки.
    """
    def __init__(self, C=1.0, method='primal', kernel='linear', gamma=None, degree=None):
        """
        C - float, коэффициент регуляризации

        method - строка, задающая решаемую задачу, может принимать значения:
            'primal' - соответствует прямой задаче
            'dual' - соответствует двойственной задаче
        kernel - строка, задающая ядро при решении двойственной задачи
            'linear' - линейное
            'polynomial' - полиномиальное
            'rbf' - rbf-ядро
        gamma - ширина rbf ядра, только если используется rbf-ядро
        d - степень полиномиального ядра, только если используется полиномиальное ядро
        Обратите внимание, что часть функций класса используется при одном методе решения,
        а часть при другом
        """
        self.C = C
        self.method = method
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree

        self.w = None
        self.w_0 = None
        self.dual_variables = None
        self.X_train = None
        self.y_train = None

        self.eps = 10 ** (-8)

    def compute_kernel_(self, X_1, X_2):
        if self.kernel == 'linear':
            return X_1.dot(X_2.T)
        elif self.kernel == 'polynomial':
            return (X_1.dot(X_2.T) + 1) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * distance_matrix(X_1, X_2) ** 2)

    def get_objective(self, X, y):
        if self.method == 'primal':
            return self.compute_primal_objective(X, y)
        elif self.method == 'dual':
            return self.compute_dual_objective(X, y)

    def compute_primal_objective(self, X, y):
        """
        Метод для подсчета целевой функции SVM для прямой задачи

        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        """
        hinge_loss = 1 - y[:, np.newaxis] * (X.dot(self.w) + self.w_0)
        hinge_loss[hinge_loss <= 0] = 0.0
        return 1 / 2 * self.w.T.dot(self.w)[0][0] + self.C * hinge_loss.mean()

    def compute_dual_objective(self, X, y):
        """
        Метод для подсчёта целевой функции SVM для двойственной задачи

        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        """
        K = self.compute_kernel_(X, X)
        yyK = y * y[:, np.newaxis] * K
        return 1 / 2 * self.dual_variables.T.dot(yyK.dot(self.dual_variables)) - \
            self.dual_variables.sum()

    def fit(self, X, y, tolerance=10**(-6), max_iter=100):
        """
        Метод для обучения svm согласно выбранной в method задаче

        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        tolerance - требуемая точность для метода обучения
        max_iter - максимальное число итераций в методе
        """
        solvers.options['reltol'] = tolerance
        solvers.options['maxiters'] = max_iter
        solvers.options['show_progress'] = False

        l = X.shape[0]
        d = X.shape[1]

        if self.method == 'primal':
            n = 1 + d + l  # for w_0, w, \xi

            P = np.zeros((n, n))
            P[np.diag_indices(1 + d)] = 1.0  # for w
            P[0, 0] = 0.0  # for w_0
            P = matrix(P)

            q = np.zeros((n, 1))
            q[1 + d:] = self.C / l  # for \xi
            q = matrix(q)

            G = np.zeros((2 * l, n))
            # soft-margin condition
            G[:l, 0] = y  # for w_0
            G[:l, 1:d + 1] = (y * X.T).T  # for w
            G[:l, d + 1:] = np.eye(l)  # for \xi
            # positive \xi (errors) condition
            G[l:, d + 1:] = np.eye(l)
            G = -G
            G = matrix(G)

            h = np.zeros((2 * l, 1))
            h[:l, 0] = -1.0  # soft-margin condition
            h[l:, 0] = 0.0  # positive \xi (errors) condition
            h = matrix(h)

            solution = np.array(solvers.qp(P, q, G, h)['x'])
            self.w = solution[1:d + 1]
            self.w_0 = solution[0]

        elif self.method == 'dual':
            P = y * y[:, np.newaxis] * self.compute_kernel_(X, X)
            P = matrix(P)

            q = -np.ones(l)
            q = matrix(q)

            G = np.zeros((2 * l, l))
            G[:l, :l] = np.eye(l)
            G[l:, :l] = -np.eye(l)
            G = matrix(G)

            h = np.zeros(l * 2)
            h[:l] = self.C / l
            h[l:] = 0.0
            h = matrix(h)

            b = matrix(np.zeros(1))

            A = np.empty((1, l))
            A[:] = y
            A = matrix(A)

            solution = np.array(solvers.qp(P, q, G, h, A, b)['x'])
            self.dual_variables = solution.ravel()
            self.X_train = X
            self.y_train = y

    def predict(self, X):
        """
        Метод для получения предсказаний на данных

        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        """
        if self.method == 'primal':
            res = X.dot(self.w) + self.w_0
        elif self.method == 'dual':
            if self.kernel == 'linear':
                if self.w is None:
                    self.w = self.get_w(self.X_train, self.y_train)
                if self.w_0 is None:
                    self.w_0 = self.get_w0(self.X_train, self.y_train)
                res = X.dot(self.w) + self.w_0
            else:
                res = self.compute_kernel_(X, self.X_train).dot(self.dual_variables * self.y_train)
        res[res >= 0] = 1
        res[res < 0] = -1
        return res.ravel()

    def get_w(self, X=None, y=None):
        """
        Получить прямые переменные (без учёта w_0)

        Если method = 'dual', а ядро линейное, переменные должны быть получены
        с помощью выборки (X, y)

        return: одномерный numpy array
        """
        if self.method == 'primal':
            return self.w
        elif self.method == 'dual':
            if self.kernel == 'linear':
                return (X.T).dot(self.dual_variables * y)

    def get_w0(self, X=None, y=None):
        """
        Получить вектор сдвига

        Если method = 'dual', а ядро линейное, переменные должны быть получены
        с помощью выборки (X, y)

        return: float
        """
        if self.method == 'primal':
            return self.w_0
        elif self.method == 'dual':
            if self.kernel == 'linear':
                mask = (self.dual_variables > self.eps)
                x_ = X[mask, :][0]
                y_ = y[mask][0]
                if self.w is None:
                    self.w = self.get_w(X, y)
                return -x_.dot(self.w) + y_

    def get_dual(self):
        """
        Получить двойственные переменные

        return: одномерный numpy array
        """
        if self.dual_variables is not None:
            return np.copy(self.dual_variables)

    def get_params(self, deep=False):
        return {
            "C": self.C
        }
