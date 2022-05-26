import numpy as np
from PARAM import SEED

np.random.seed(SEED)


class Solver:
    def __init__(self, M, N):
        self.M = M
        self.N = N
        self.A = np.random.rand(M, N)
        self.opt_w = np.random.rand(N, 1)
        self.b = self.A @ self.opt_w + np.random.rand(M, 1)  # A @ w + error
        self.init_w = np.random.rand(N, 1)

    def steepest_descent_method(self, lam, max_iter=20):
        w = self.init_w
        error = []
        y = []
        for k in range(max_iter + 1):
            step_size = 1 / np.abs(np.max(np.linalg.eig(self.ddf(w, lam))[0]))
            error.append(np.linalg.norm(w - self.opt_w))
            y.append(self.f(w, lam))
            w = w - self.df(w, lam) * step_size
        return error, y

    def steepest_descent_method_with_armijo_rule(self, lam, max_iter=20, alpha=1, xi=1e-3, tau=0.5):
        w = self.init_w
        step_size = alpha
        error = []
        y = []
        for k in range(max_iter + 1):
            while self.f(w - step_size * self.df(w, lam), lam) > \
                    self.f(w, lam) - xi * step_size * self.df(w, lam).T @ self.df(w, lam):
                step_size *= tau
            error.append(np.linalg.norm(w - self.opt_w))
            y.append(self.f(w, lam))
            w = w - self.df(w, lam) * step_size
        return error, y

    def nesterovs_accelerated_gradient_algorithm(self, lam, max_iter=20):
        w = self.init_w
        w_bf = self.init_w
        beta = 0
        error = []
        y = []
        for k in range(max_iter + 1):
            step_size = 1 / np.abs(np.max(np.linalg.eig(self.ddf(w, lam))[0]))
            error.append(np.linalg.norm(w - self.opt_w))
            y.append(self.f(w, lam))
            (w, w_bf) = (w + beta * (w - w_bf) - step_size * self.df(w + beta * (w - w_bf), lam), w.copy())
            beta = k / (k + 3)
        return error, y

    def f(self, w, lam):
        return np.linalg.norm(self.b - self.A @ w) ** 2 + lam * np.linalg.norm(w) ** 2

    def df(self, w, lam):
        return 2 * (self.A.T @ self.A @ w - self.A.T @ self.b + lam * w)

    def ddf(self, w, lam):
        return 2 * (self.A.T @ self.A + lam * np.eye(self.N))
