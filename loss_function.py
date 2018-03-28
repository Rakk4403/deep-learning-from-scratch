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
