import matplotlib.pyplot as plt
from solver import Solver
from function import MyFunction


def ex2():
    m = 20
    n = 50
    max_iter = 50
    obj = MyFunction(m, n)
    slv = Solver(obj)
    x = [k for k in range(max_iter + 1)]
    lams = [0, 1, 10]
    error1 = []
    error2 = []
    for lam in lams:
        er1, y1 = slv.steepest_descent_method(lam, max_iter=max_iter)
        er2, y2 = slv.steepest_descent_method_with_armijo_rule(lam, max_iter=max_iter)
        error1.append(er1)
        error2.append(er2)
        plt.title("λ = {}".format(lam))
        plt.plot(x[1:], y1[1:], label="steepest")
        plt.plot(x[1:], y2[1:], label="armijo")
        plt.legend()
        plt.show()

    plt.title("steepest descent method")
    for lam, er1 in zip(lams, error1):
        plt.plot(x, er1, label="λ = {}".format(lam))
    plt.legend()
    plt.show()

    plt.title("steepest descent method with armijo rule")
    for lam, er2 in zip(lams, error2):
        plt.plot(x, er2, label="λ = {}".format(lam))
    plt.legend()
    plt.show()
