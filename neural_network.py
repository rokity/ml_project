from layer import *
from functions_factory import FunctionsFactory
from kernel_initialization import *
import matplotlib.pyplot as plt
import sys


class NeuralNetwork:

    def __init__(self, loss, metric):
        """

        @param loss: string that represents the loss function to use
        @param metric: string that represents the metric to use
        """
        self.layers = []
        self.loss = FunctionsFactory.build(loss)
        self.metric = FunctionsFactory.build(metric)
        self.history = dict()

    def compile(self, lr=1e-3, momentum=0.0, l2=0.0):
        """

        @param lr: learning rate
        @param momentum: momentum
        @param l2: l2 regularizer hyperparameter
        @return:
        """
        self.lr = lr
        self.momentum = momentum
        self.l2 = l2
        for i in range(len(self.layers)):
            self.layers[i].compile()

    def add_layer(self, dim_out, input_dim=None, activation='linear', kernel_initialization=RandomUniformInitialization(), label='Dense input'):
        """

        @param dim_out: dimension of the output
        @param input_dim: dimension of the input
        @param activation: string that represents the activation function
        @param kernel_initialization: weights initialization
        @param label: layer name
        @return:
        """
        f_act = FunctionsFactory.build(activation)
        if input_dim is not None:
            layer = Layer(input_dim, dim_out, f_act, self.loss, kernel_initialization, label)
        else:
            layer = Layer(self.layers[-1].dim_out, dim_out, f_act, self.loss, kernel_initialization, label)
        self.layers.append(layer)

    def add_output_layer(self, dim_out, input_dim=None, activation='linear', kernel_initialization=RandomUniformInitialization(), label='Dense output'):
        """

        @param dim_out: dimension of the output
        @param input_dim: dimension of the input
        @param activation: string that represents the activation function
        @param kernel_initialization: weights initialization
        @param label: layer name
        @return:
        """
        f_act = FunctionsFactory.build(activation)
        if input_dim is not None:
            layer = OutputLayer(input_dim, dim_out, f_act, self.loss, kernel_initialization, label)
        else:
            layer = OutputLayer(self.layers[-1].dim_out, dim_out, f_act, self.loss, kernel_initialization, label)
        self.layers.append(layer)

    def predict(self, X: np.ndarray):
        """

        @param X: input samples
        @return: output predictions
        """
        out = np.zeros((X.shape[0], self.layers[-1].dim_out))
        n_samples = X.shape[0]
        for i in range(n_samples):
            x = X[i]
            out[i] = self.__feedforward(x)
        return out

    def evaluate(self, X: np.ndarray, Y: np.ndarray):
        """

        @param X: input samples
        @param Y: output targets
        @return: loss and metric value
        """
        err = 0
        metric = 0
        n_samples = X.shape[0]
        for i in range(n_samples):
            x = X[i].reshape((1, X.shape[1]))
            d = Y[i].reshape((1, Y.shape[1]))
            y = self.__feedforward(x)
            err += float(self.loss.compute_fun(d.T, y))
            metric += float(self.metric.compute_fun(d.T, y))
        return err / n_samples, metric / n_samples

    def __feedforward(self, x: np.ndarray):
        """

        @param x: input sample
        @return: output prediction
        """
        x = x.reshape((x.shape[1], x.shape[0]))
        for layer in self.layers:
            x = layer.feedforward(x)
        return x

    def backpropagation(self, d):
        """

        @param d: real output
        execute backpropagation algorithm
        """
        loc_grad = None
        for layer in reversed(self.layers):
            loc_grad = layer.backpropagation(loc_grad, d)

    def update_weights(self, batch_size):
        """

        @param batch_size: size of the batch
        update the parameters of the model
        """
        for layer in self.layers:
            layer.update_weights(self.lr, self.momentum, self.l2, batch_size)

    def fit(self, X_train, Y_train, epochs, batch_size, vl=None, ts=None, tol=None, shuffle=False, verbose=False):
        """

        @param X_train: training samples
        @param Y_train: training targets
        @param epochs: epochs number
        @param batch_size: size of the batch
        @param vl: pair (validation samples, validation targets)
        @param ts: pair (test samples, test targets) (used only for plot)
        @param tol: tolerance on the training set
                    if null => It uses an early stopping method
        @param shuffle: True if you want shuffle training data (at each epoch),
                        False otherwise
        @param verbose: used to print some informations
        @return: error on the validation set (if it's not None) otherwise error on the training set
        """

        self.history[self.loss.name] = []
        self.history[self.metric.name] = []

        if vl is not None:
            X_val, Y_val = vl
            self.history["val_" + self.loss.name] = []
            self.history["val_" + self.metric.name] = []

        if ts is not None:
            X_test, Y_test = ts
            self.history["test_" + self.loss.name] = []
            self.history["test_" + self.metric.name] = []

        n_samples = X_train.shape[0]
        curr_epoch = 0
        curr_i = 0
        tr_err_pen = 0
        vl_err = 0
        min_tr_err = sys.float_info.max
        min_vl_err = sys.float_info.max

        n_batch = int(n_samples / batch_size)
        tr_loss_batch = np.zeros(n_batch)
        tr_lossr_batch = np.zeros(n_batch)
        tr_metric_batch = np.zeros(n_batch)

        while curr_epoch < epochs:

            if shuffle:
                idx = np.random.permutation(n_samples)
                X_train = X_train[idx]
                Y_train = Y_train[idx]

            for nb in range(n_batch):
                loc_err = 0
                loc_metric = 0

                for _ in range(batch_size):
                    x = X_train[curr_i].reshape(1, X_train.shape[1])
                    d = Y_train[curr_i].reshape(1, Y_train.shape[1])
                    y = self.__feedforward(x)
                    self.backpropagation(d.T)
                    loc_err += self.loss.compute_fun(d.T, y)
                    loc_metric += self.metric.compute_fun(d.T, y)
                    curr_i = (curr_i + 1) % n_samples

                tr_loss_batch[nb] = loc_err / batch_size
                tr_lossr_batch[nb] = (loc_err + self.l2 * self.sum_square_weights()) / batch_size
                tr_metric_batch[nb] = loc_metric / batch_size

                self.update_weights(batch_size)

            # compute average loss/metric in training set
            tr_err = np.mean(tr_loss_batch)
            tr_err_pen = np.mean(tr_lossr_batch)
            tr_metric = np.mean(tr_metric_batch)

            self.history[self.loss.name].append(tr_err)
            self.history[self.metric.name].append(tr_metric)

            min_tr_err = min(min_tr_err, tr_err_pen)

            # compute error in validation set
            if vl is not None:
                vl_err, vl_metric = self.evaluate(X_val, Y_val)
                self.history["val_" + self.loss.name].append(vl_err)
                self.history["val_" + self.metric.name].append(vl_metric)
            else:   # if there is no validation set it uses the training for the early stopping
                vl_err = tr_err_pen

            # generalization loss
            gl = 100 * ((vl_err / min_vl_err) - 1)

            min_vl_err = min(min_vl_err, vl_err)

            curr_epoch += 1

            # compute error in test set
            if ts is not None:
                ts_err, ts_metric = self.evaluate(X_test, Y_test)
                self.history["test_" + self.loss.name].append(ts_err)
                self.history["test_" + self.metric.name].append(ts_metric)

            if verbose:
                print(
                    "It {:6d}: tr_err (with penality term): {:.6f},"
                    "\t tr_err (without penality term): {:.6f}"
                    "\t vl_err: {:.6f},".format(
                        curr_epoch,
                        tr_err_pen,
                        tr_err,
                        self.history[self.loss.name][-1]),
                    end='\r'
                )

            if tol is None:
                if gl > 0.2 and tr_err_pen - min_tr_err > 0:
                    break
            else:
                if tr_err_pen < tol:
                    break

        self.history["epochs"] = list(range(curr_epoch))

        if verbose:
            print()
            print("Exit at epoch: {}".format(curr_epoch))

            print("{} training set: {:.6f}".format(self.loss.name, self.history[self.loss.name][-1]))
            if vl is not None:
                print("{} validation set: {:.6f}".format(self.loss.name, self.history["val_" + self.loss.name][-1]))
            if ts is not None:
                print("{} test set: {:.6f}".format(self.loss.name, self.history["test_" + self.loss.name][-1]))
            if self.metric.name == 'accuracy':
                print("% accuracy training set: {}".format(self.history[self.metric.name][-1] * 100))
                if vl is not None:
                    print("% accuracy validation set: {}".format(self.history["val_" + self.metric.name][-1] * 100))
                if ts is not None:
                    print("% accuracy test set: {}".format(self.history["test_" + self.metric.name][-1] * 100))
            else:
                print("{} training set: {:.6f}".format(self.metric.name, self.history[self.metric.name][-1]))
                if vl is not None:
                    print("{} validation set: {:.6f}".format(self.metric.name, self.history["val_" + self.metric.name][-1]))
                if ts is not None:
                    print("{} accuracy test set: {:.6f}".format(self.metric.name, self.history["test_" + self.metric.name][-1]))

        return vl_err

    def sum_square_weights(self):
        """

        @param size: dimension of the batch

        It's used to regularization
        """
        sum = 0
        for layer in self.layers:
            sum += np.sum(np.square(layer.w))
        return sum

    def plot_loss(self, val=False, test=False, show=True, path=None):
        epochs = self.history['epochs']
        plt.xlabel('epochs')
        plt.ylabel(self.loss.name)
        plt.plot(epochs, self.history[self.loss.name], 'r', label="Training")
        if val:
            plt.plot(epochs, self.history["val_" + self.loss.name], 'g--', label="Validation")
        if test:
            plt.plot(epochs, self.history["test_" + self.loss.name], 'b-.', label="Test")
        plt.legend()
        if path is not None:
            plt.draw()
            plt.savefig(path)
        if show:
            plt.show()
        plt.close()
        

    def plot_metric(self, val=False, test=False, show=True, path=None):
        epochs = self.history['epochs']
        plt.xlabel('epochs')
        plt.ylabel(self.metric.name)
        plt.plot(epochs, self.history[self.metric.name], 'r', label="Training")
        if val:
            plt.plot(epochs, self.history["val_" + self.metric.name], 'g--', label="Validation")
        if test:
            plt.plot(epochs, self.history["test_" + self.metric.name], 'b-.', label="Test")
        plt.legend()
        if path is not None:
            plt.draw()
            plt.savefig(path)
        if show:
            plt.show()
        plt.close()
        
