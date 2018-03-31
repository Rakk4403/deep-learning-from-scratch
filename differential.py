# author: rakk4403
import numpy as np
import matplotlib.pylab as plt


def numerical_diff_badcase(f, x):
    """
    It would be rounding error, if input is too small.
    """
    h = 10e-50  # close to 0
    return (f(x + h) - f(x)) / h


def numerical_diff(f, x):
    # Right function
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


def function_1(x):
    # temporary function for differential target
    return 0.01 * (x ** 2) + 0.1 * x


if __name__ == '__main__':
    # check function_1
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)

    # differential of function_1
    plt.plot(x, numerical_diff(function_1, x))
    plt.show()
