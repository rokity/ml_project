import numpy as np
from kernel_initialization import *


class Layer:
    def __init__(self, dim_in, dim_out, f_act, loss, kernel_initialization, name):
        """

        @param dim_in: input dimension
        @param dim_out: output dimension
        @param f_act: activation function
        @param loss: loss fucntion
        @param kernel_initialization: weights initialization
        @param name: layer name
        """
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.f_act = f_act
        self.loss = loss
        self.name = name
        self.kernel_initialization = kernel_initialization

    def compile(self):
        """

        initialize the layer
        """
        self.__init_layer(self.kernel_initialization)

    def __init_layer(self, kernel_initialization):
        """

        @param kernel_initialization: weights initialization

        initialize the layer
        """
        self.w = kernel_initialization.initialize(self.dim_out, self.dim_in)
        self.b = np.zeros((self.dim_out, 1))
        self.delta_w = np.zeros(self.w.shape)
        self.prev_delta_w = np.zeros(self.w.shape)
        self.delta_b = np.zeros(self.b.shape)
        self.prev_delta_b = np.zeros(self.b.shape)
        self.v = None
        self.y = None
        self.x = None

    def feedforward(self, x):
        """

        @param x: input sample
        @return: output prediction
        """
        self.x = x.copy()
        self.v = np.dot(self.w, x) + self.b
        self.y = self.f_act.compute_fun(self.v)
        return self.y

    def backpropagation(self, loc_grad, d):
        """

        @param loc_grad: gradient of the next layer
        @param d: real output (here not used @see OutputLayer)
        @return: local gradient (to be used in the previous layer)
        """
        partial = loc_grad * self.f_act.compute_der(self.v)
        self.delta_w += np.dot(partial, self.x.T)
        return np.dot(self.w.T, partial)

    def update_weights(self, eta, alpha, lam, batch_size):
        """

        @param eta: learning rate
        @param alpha: momentum
        @param lam: regularizer
        @param batch_size: size of the batch
        @return:
        """

        self.delta_w = eta*self.delta_w/batch_size + alpha*self.prev_delta_w
        self.delta_b = eta*self.delta_b/batch_size + alpha*self.prev_delta_b
        self.w -= self.delta_w
        self.w -= (lam*self.w/batch_size)
        self.b += self.delta_b
        self.prev_delta_w = self.delta_w.copy()
        self.prev_delta_b = self.delta_b.copy()
        self.delta_w = np.zeros(self.w.shape)
        self.delta_b = np.zeros(self.b.shape)

    def print_info(self):
        """

        prints layer information
        """
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
        """

        @param loc_grad: gradient of the next layer
        @param d: real output
        @return: local gradient (to be used in the previous layer)
        """
        '''
        loc_grad = self.loss.compute_der(d, self.y).dot(self.f_act.compute_der(self.v))
        self.delta_w += np.dot(self.x.T, loc_grad)
        '''

        partial = self.loss.compute_der(d, self.y)*self.f_act.compute_der(self.v)
        self.delta_w += np.dot(partial, self.x.T)
        return np.dot(self.w.T, partial)
