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


def function_2(x):
    # temporary function for partial differenctials
    return np.sum(x ** 2)  # or x[0]**2 + x[1]**2


def numerical_gradient(f, x):
    """
    gradient is same as partial differentials
    """
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    f: target function to optimize
    init_x: initial value
    lr: learning rate
    step_num: repeat count for descent method
    """
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


if __name__ == '__main__':
    # check function_1
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.show()

    # differential of function_1
    ret = numerical_diff(function_1, 5)
    print(ret)

    # partial differentials (gradient)
    ret = numerical_gradient(function_2, np.array([3.0, 4.0]))
    print(ret)
    ret = numerical_gradient(function_2, np.array([0.0, 2.0]))
    print(ret)
    ret = numerical_gradient(function_2, np.array([3.0, 0.0]))
    print(ret)

    # gradient_descent
    init_x = np.array([-3.0, 4.0])
    ret = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
    print('gradient descent: {}'.format(ret))