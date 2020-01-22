import numpy as np


class Layer:
    def __init__(self, dim_in, dim_out, f_act, loss, fan_in, name):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.f_act = f_act
        self.loss = loss
        self.name = name
        self.__init_layer(fan_in)

    def __init_layer(self, fan_in):
        max = 0.7 * 2 / fan_in
        min = - max
        self.w = (np.random.randn(self.dim_in, self.dim_out) * (max - min)) + min
        self.b = np.random.randn(1, self.dim_out)
        self.delta_w = np.zeros((self.dim_in, self.dim_out))
        self.prev_delta_w = np.zeros((self.dim_in, self.dim_out))
        self.delta_b = np.zeros((1, self.dim_out))
        self.prev_delta_b = np.zeros((1, self.dim_out))
        self.v = None
        self.y = None
        self.x = None

    def feedforward(self, x):
        self.x = x.copy()
        self.v = np.dot(x, self.w) + self.b
        self.y = self.f_act.compute_fun(self.v)
        return self.y

    def backpropagation(self, loc_grad, d):
        loc_grad = loc_grad * self.f_act.compute_der(self.v)
        self.delta_w += np.dot(self.x.T, loc_grad)
        self.delta_b += loc_grad
        return np.dot(loc_grad, self.w.T)

    def update_weights(self, eta, alpha, lam, batch_size):
        self.delta_w = -eta*self.delta_w/batch_size + alpha*self.prev_delta_w
        self.delta_b = -eta*self.delta_b/batch_size + alpha*self.prev_delta_b
        self.w += self.delta_w + lam*self.w/batch_size
        self.b += self.delta_b
        self.prev_delta_w = self.delta_w.copy()
        self.prev_delta_b = self.delta_b.copy()
        self.delta_w = np.zeros((self.dim_in, self.dim_out))
        self.delta_b = np.zeros((1, self.dim_out))

    def print_info(self):
        print('name: {}\n'
              'input size: {}\n'
              'output size: {}\n'
              'activation function: {}\n'
              'loss function: {}\n'
              '--------'
              .format(self.name, self.dim_in, self.dim_out, self.f_act.name, self.loss.name)
              )


class OutputLayer(Layer):

    def backpropagation(self, loc_grad, d):
        loc_grad = self.loss.compute_der(d, self.y) * self.f_act.compute_der(self.v)
        self.delta_w += np.dot(self.x.T, loc_grad)
        return np.dot(loc_grad, self.w.T)
