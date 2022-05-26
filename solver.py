import numpy as np
from PARAM import SEED
from function import ObjectiveFunction

np.random.seed(SEED)


class Solver:
    def __init__(self, function: ObjectiveFunction):
        self.objective_function = function
        self.init_w = function.init_w
        self.opt_w = function.opt_w
        self.f = function.f
        self.df = function.df
        self.ddf = function.ddf

    def steepest_descent_method(self, lam, max_iter=20):
        w = self.init_w
        step_size = 1 / np.abs(np.max(np.linalg.eig(self.ddf(w, lam))[0]))
        error = []
        y = []
        for k in range(max_iter + 1):
            error.append(self.calc_error(w))
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
                    self.f(w, lam) - xi * step_size * self.df(w, lam).T @ \
                    self.df(w, lam):
                step_size *= tau
            error.append(self.calc_error(w))
            y.append(self.f(w, lam))
            w = w - self.df(w, lam) * step_size
        return error, y

    def nesterovs_accelerated_gradient_algorithm(self, lam, max_iter=20):
        w = self.init_w
        w_bf = self.init_w
        step_size = 1 / np.abs(np.max(np.linalg.eig(self.ddf(w, lam))[0]))
        beta = 0
        error = []
        y = []
        for k in range(max_iter + 1):
            error.append(self.calc_error(w))
            y.append(self.f(w, lam))
            (w, w_bf) = (w + beta * (w - w_bf) - step_size * self.df(w + beta * (w - w_bf), lam), w.copy())
            beta = k / (k + 3)
        return error, y

    def calc_error(self, w):
        if self.opt_w is not None:
            return np.linalg.norm(w - self.opt_w)
        else:
            return None
