import numpy as np


class AbstractKernelInitialization:

    def initialize(self, dim_in, dim_out):
        pass


class RandomInitialization(AbstractKernelInitialization):
    def __init__(self, trsl=1.0):
        self.trsl = trsl

    def initialize(self, dim_in, dim_out):
        return np.random.randn(dim_in, dim_out) * self.trsl


class RandomNormalInitialization(AbstractKernelInitialization):
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std

    def initialize(self, dim_in, dim_out):
        return np.random.normal(self.mean, self.std, (dim_in, dim_out))


class RandomUniformInitialization(AbstractKernelInitialization):
    def __init__(self, low=-0.05, high=0.05):
        self.low = low
        self.high = high

    def initialize(self, dim_in, dim_out):
        return np.random.uniform(self.low, self.high, (dim_in, dim_out))


class HeInitialization(AbstractKernelInitialization):
    def __init__(self, size_l):
        self.size_l = size_l

    def initialize(self, dim_in, dim_out):
        return np.random.randn(dim_in, dim_out) * np.sqrt(2/self.size_l)


class XavierUniformInitialization(AbstractKernelInitialization):
    def __init__(self):
        pass

    def initialize(self, dim_in, dim_out):
        limit = np.sqrt(6/(dim_in + dim_out))
        return np.random.uniform(-limit, limit, (dim_in, dim_out))


class XavierNormalInitialization(AbstractKernelInitialization):
    def __init__(self):
        pass

    def initialize(self, dim_in, dim_out):
        std = np.sqrt(2/(dim_in + dim_out))
        return np.random.normal(0, std, (dim_in, dim_out))


class ZerosInitialization(AbstractKernelInitialization):
    def __init__(self):
        pass

    def initialize(self, dim_in, dim_out):
        return np.zeros((dim_in, dim_out))

