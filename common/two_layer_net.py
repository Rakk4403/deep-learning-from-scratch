# author: rakk4403
import numpy as np

from differential import numerical_gradient
from loss_function import cross_entropy_error_for_minibatch
from output_layer import softmax
from step_function import sigmoid


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):
        # initiate weights
        self.params = {
            'W1': weight_init_std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'W2': weight_init_std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size),
        }

    def predict(self, x):
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error_for_minibatch(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        def loss_W():
            return self.loss(x, t)

        grads = {
            'W1': numerical_gradient(loss_W, self.params['W1']),
            'b1': numerical_gradient(loss_W, self.params['b1']),
            'W2': numerical_gradient(loss_W, self.params['W2']),
            'b2': numerical_gradient(loss_W, self.params['b2'])
        }
        return grads


if __name__ == '__main__':
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params['W1'].shape)  # (784, 100)
    print(net.params['b1'].shape)  # (100, )
    print(net.params['W2'].shape)  # (100, 10)
    print(net.params['b2'].shape)  # (10, )
