# author: rakk4403
import numpy as np


def mean_squared_error(y, t):
    """
    y: np.array
    t: np.array
    """
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    """
    y : np.array
    t : np.array
    """
    # to prevent 0 for log result
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def cross_entropy_error_for_minibatch(y, t, one_hot_encoding=False):
    """
    y: np.array, neural network output
    t: np.array, test target
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    if one_hot_encoding:
        return -np.sum(t * np.log(y)) / batch_size
    else:
        return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


if __name__ == '__main__':
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    print(cross_entropy_error(np.array(y), np.array(t)))

    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    print(cross_entropy_error(np.array(y), np.array(t)))
