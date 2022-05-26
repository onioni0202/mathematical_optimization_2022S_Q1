import numpy as np


class ObjectiveFunction:
    def __init__(self):
        self.opt_w = None
        self.init_w = None  # Initial value

    def f(self, w, lam):
        raise NotImplementedError()

    def df(self, w, lam):
        raise NotImplementedError()

    def ddf(self, w, lam):
        raise NotImplementedError()


class MyFunction(ObjectiveFunction):
    def __init__(self, m, n):
        super(MyFunction, self).__init__()
        self.M = m
        self.N = n
        self.A = np.random.rand(m, n)
        self.opt_w = np.random.rand(n, 1)  # Answer
        self.b = self.A @ self.opt_w + np.random.rand(m, 1)  # A @ w + error
        self.init_w = np.random.rand(n, 1)  # Initial value

    def f(self, w, lam):
        return np.linalg.norm(self.b - self.A @ w) ** 2 + lam * np.linalg.norm(w) ** 2

    def df(self, w, lam):
        return 2 * (self.A.T @ self.A @ w - self.A.T @ self.b + lam * w)

    def ddf(self, w, lam):
        return 2 * (self.A.T @ self.A + lam * np.eye(self.N))
