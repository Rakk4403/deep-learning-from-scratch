import numpy as np
from .output_layer import softmax
from .loss_function import cross_entropy_error_for_minibatch


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # init with regular distribution

    def predict(self, x):
        """
        x: np.array, input data
        """
        return np.dot(x, self.W)

    def loss(self, x, t):
        """
        x: np.array, input data
        t: np.array, solution labels
        """
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error_for_minibatch(y, t)

        return loss
