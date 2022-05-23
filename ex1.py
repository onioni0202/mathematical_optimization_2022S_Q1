import matplotlib.pyplot as plt
from solver import Solver


def ex1():
    M = 20
    N = 50
    max_iter = 50
    slv = Solver(M, N)
    x = [k for k in range(max_iter + 1)]
    errors = []
    lams = [0, 1, 10]
    for lam in lams:
        error, y = slv.steepest_descent_method(lam, max_iter)
        errors.append(error)
        plt.plot(x[1:], y[1:], label="λ = {}".format(lam))
        plt.legend()
        plt.show()
    for lam, error in zip(lams, errors):
        plt.plot(x, error, label="λ = {}".format(lam))
    plt.legend()
    plt.show()
