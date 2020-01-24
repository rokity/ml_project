import numpy as np


class FunctionsFactory:

    @staticmethod
    def build(name):
        if name == 'tanh':
            return Function(name, tanh, tanh_der)
        elif name == 'sigmoid':
            return Function(name, sigmoid, sigmoid_der)
        elif name == 'linear':
            return Function(name, linear, linear_der)
        elif name == 'lms':
            return Function(name, lms, lms_der)
        elif name == 'eucledian':
            return Function(name, euclidean, euclidean_der)
        elif name == 'accuracy':
            return Function(name, accuracy, None)
        elif name == 'accuracy_multiple':
            return Function(name, accuracy_multiple, None)


class Function:
    def __init__(self, name, c_fun, c_der):
        self.name = name
        self.compute_fun = c_fun
        self.compute_der = c_der


# ----------------- Activation functions -----------------

def tanh(x):
    return np.tanh(x)


def tanh_der(x):
    return 1 - tanh(x)**2


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def linear(x):
    return x


def linear_der(x):
    return np.identity(x.size)


# ----------------- Loss functions -----------------

def lms(d, y):
    return np.dot(d - y, d - y)


def lms_der(d, y):
    return -2*(d - y)


def euclidean(d, y):
    return np.sqrt(np.dot(d - y, d - y))


def euclidean_der(d, y):
    return -(d - y) / np.sqrt(np.dot(d - y, d - y))


# ----------------- Accuracy functions -----------------

def accuracy(d, y):
    if np.abs(d - y) < 0.5:
        return 1
    else:
        return 0


def accuracy_multiple(d, y):
    if np.argmax(d) == np.argmax(y):
        return 1
    else:
        return 0