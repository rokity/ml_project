import numpy as np

from numba import jit

class FunctionsFactory:

    @staticmethod
    def build(name):
        if name == 'tanh':
            return Function(name, tanh, tanh_der)
        elif name == 'sigmoid':
            return Function(name, sigmoid, sigmoid_der)
        elif name == 'reLU':
            return Function(name, reLU, reLU_der)
        elif name == 'linear':
            return Function(name, linear, linear_der)
        elif name == 'lms':
            return Function(name, lms, lms_der)
        elif name == 'mee':
            return Function(name, mee, mee_der)
        elif name == 'mse':
            return Function(name, mse, mse_der)
        elif name == 'accuracy':
            return Function(name, accuracy, None)



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


@jit
def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def linear(x):
    return x


def linear_der(x):
    return np.identity(x.size)


def reLU(x):
    x[x <= 0] = 0.0
    return x


def reLU_der(x):
    x[x > 0] = 1.0
    x[x <= 0] = 0.0
    return np.diag(x).reshape((x.shape[0], x.shape[0]))


# ----------------- Loss functions -----------------

def lms(d, y):
    return (d - y)**2


def lms_der(d, y):
    return -2*(d - y)

@jit
def mse(d, y):
    diff = d - y
    return np.dot(diff, diff.T)

@jit
def mse_der(d, y):
    return -2*(d - y)

@jit
def mee(d, y):
    return np.linalg.norm(d - y)

@jit
def mee_der(d, y):
    return -(d - y) / np.linalg.norm(d - y)


# ----------------- Accuracy functions -----------------

def accuracy(d, y):
    if np.abs(d - y) < 0.5:
        return 1
    else:
        return 0