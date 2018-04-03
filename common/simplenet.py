import numpy as np
from .output_layer import softmax
from .loss_function import cross_entropy_error_for_minibatch
from .differential import numerical_gradient


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


if __name__ == '__main__':
    net = SimpleNet()
    print(net.W)  # weight parameters

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))  # index of max value

    t = np.array([0, 0, 1])  # labels
    print(net.loss(x, t))

    def f(W):
        return net.loss(x, t)

    dW = numerical_gradient(f, net.W)
    print(dW)
