import matplotlib.pyplot as plt
from solver import Solver
from function import MyFunction


def ex1():
    m = 20
    n = 50
    max_iter = 50
    obj = MyFunction(m, n)
    slv = Solver(obj)
    x = [k for k in range(max_iter + 1)]
    errors = []
    lams = [0, 1, 10]
    for lam in lams:
        error, y = slv.steepest_descent_method(lam, max_iter)
        errors.append(error)
        plt.title("λ = {}".format(lam))
        plt.plot(x[1:], y[1:], label="steepest".format(lam))
        plt.legend()
        plt.show()

    plt.title("steepest descent method")
    for lam, error in zip(lams, errors):
        plt.plot(x, error, label="λ = {}".format(lam))
    plt.legend()
    plt.show()
