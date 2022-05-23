import matplotlib.pyplot as plt
from solver import Solver


def ex2():
    M = 20
    N = 50
    max_iter = 50
    slv = Solver(M, N)
    x = [k for k in range(max_iter + 1)]
    lams = [0, 1, 10]
    for lam in lams:
        _, y1 = slv.steepest_descent_method(lam, max_iter=max_iter)
        _, y2 = slv.steepest_descent_method_with_armijo_rule(lam, max_iter=max_iter)
        plt.title("λ = {}".format(lam))
        plt.plot(x[1:], y1[1:], label="step size: 1/L")
        plt.plot(x[1:], y2[1:], label="step size: armijo")
        plt.legend()
        plt.show()
