from layer import *
from dataset import Dataset
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

    def compile(self, lr=0.1, momentum=0.0, l2=0.0):
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

    def add_input_layer(self, dim_in, dim_out, f_act, kernel_initialization=RandomUniformInitialization(), label='Input layer'):
        """

        @param dim_in: dimension of the input
        @param dim_out: dimension of the output
        @param f_act: string that represents the activation function
        @param kernel_initialization: weights initialization
        @param label: layer name
        @return:
        """
        f_act = FunctionsFactory.build(f_act)
        layer = Layer(dim_in, dim_out, f_act, self.loss, kernel_initialization, label)
        self.layers.append(layer)

    def add_hidden_layer(self, dim_out, f_act, kernel_initialization=RandomUniformInitialization(), label='Hidden layer'):
        """

        @param dim_out: dimension of the output
        @param f_act: string that represents the activation function
        @param kernel_initialization: weights initialization
        @param label: layer name
        @return:
        """
        f_act = FunctionsFactory.build(f_act)
        layer = Layer(self.layers[-1].dim_out, dim_out, f_act, self.loss, kernel_initialization, label)
        self.layers.append(layer)

    def add_output_layer(self, dim_out, f_act, kernel_initialization=RandomUniformInitialization(), label='Input layer'):
        """

        @param dim_out: dimension of the output
        @param f_act: string that represents the activation function
        @param kernel_initialization: weights initialization
        @param label: layer name
        @return:
        """
        f_act = FunctionsFactory.build(f_act)
        layer = OutputLayer(self.layers[-1].dim_out, dim_out, f_act, self.loss, kernel_initialization, label)
        self.layers.append(layer)

    def predict_sample(self, x: np.ndarray):
        """

        @param x: input sample
        @return: output prediction
        """
        return self.__feedforward(x)

    def predict(self, X: np.ndarray):
        """

        @param X: input samples
        @return: output predictions
        """
        out = np.zeros((X.shape[0], self.layers[-1].dim_out))
        for i in range(X.shape[0]):
            x = X[i]
            x = x.reshape(X.shape[0], 1)
            out[i] = self.__feedforward(x.T)
        return out

    def predict_dataset(self, dataset: Dataset):
        """

        @param dataset: dataset with samples and targets
        @return: output predictions
        """
        out = np.zeros((dataset.size, dataset.dim_out))
        for i in range(dataset.size):
            x, d = dataset.get_data(i)
            x = x.reshape(x.shape[0], 1)
            out[i] = self.__feedforward(x.T)
        return out

    def __feedforward(self, x: np.ndarray):
        """

        @param x: input sample
        @return: output prediction
        """
        for layer in self.layers:
            x = layer.feedforward(x)
        return x

    def evaluate_dataset(self, dataset: Dataset):
        """

        @param dataset: input samples
        @return: loss and metric values
        """
        err = 0
        metric = 0
        for i in range(dataset.size):
            x, d = dataset.get_data(i)
            x = x.reshape(x.shape[0], 1)
            y = self.__feedforward(x.T)
            err += self.loss.compute_fun(d, y)
            metric += self.metric.compute_fun(d, y)
        return float(err) / dataset.size, metric / dataset.size

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

        @return: update the parameters of the model
        """
        for layer in self.layers:
            layer.update_weights(self.lr, self.momentum, self.l2, batch_size)

    def fit(self, tr, epochs=1000, batch_size=32, vl=None, ts=None, tol=None, verbose=False):
        """

        @param epochs: epochs number
        @param tr: training set
        @param vl: validation set
        @param ts: test set (used only for plot)
        @param batch_size: size of the batch
        @param verbose: used to print some informations
        @param tol: tolerance on the training set
                    if null => It uses an early stopping method
        @return: error on the validation set (if it's not None) otherwise error on the training set
        """
        self.history[self.loss.name] = []
        self.history[self.metric.name] = []
        if vl is not None:
            self.history["val_" + self.loss.name] = []
            self.history["val_" + self.metric.name] = []

        if ts is not None:
            self.history["test_" + self.loss.name] = []
            self.history["test_" + self.metric.name] = []

        it = 0
        curr_i = 0
        training_err = 0
        validation_err = 0
        min_tr_err = sys.float_info.max
        min_vl_err = sys.float_info.max

        while it < epochs:

            # train
            for _ in range(batch_size):
                x, d = tr.get_data(curr_i)
                x = x.reshape(x.shape[0], 1)
                d = d.reshape(d.shape[0], 1)
                y = self.__feedforward(x.T)
                self.backpropagation(d.T)
                curr_i = (curr_i + 1) % tr.size

            # compute error in training set
            err, metric = self.evaluate_dataset(tr)
            err = err

            self.history[self.loss.name].append(err)
            self.history[self.metric.name].append(metric)

            tot_weights = self.sum_square_weights(tr.size)

            training_err = err + (self.l2 * tot_weights)

            min_tr_err = min(min_tr_err, training_err)

            # compute error in validation set
            if vl is not None:
                err, metric = self.evaluate_dataset(vl)
                self.history["val_" + self.loss.name].append(err)
                self.history["val_" + self.metric.name].append(metric)
                validation_err = err
            else:   # if there is no validation set it uses the training for the early stopping
                validation_err = training_err

            # generalization loss
            gl = 100 * ((validation_err / min_vl_err) - 1)

            min_vl_err = min(min_vl_err, validation_err)

            it += 1

            # compute error in test set
            if ts is not None:
                err, metric = self.evaluate_dataset(ts)
                self.history["test_" + self.loss.name].append(err)
                self.history["test_" + self.metric.name].append(metric)

            self.update_weights(batch_size)

            if verbose:
                print(
                    "It {:6d}: tr_err (with penality term): {:.6f},\t vl_err: {:.6f},\t gl: {:.6f}".format(it, training_err, validation_err, gl),
                    end='\r'
                )

            if tol is None:
                if gl > 0.2 and training_err - min_tr_err > 0:
                    break
            else:
                if training_err < tol:
                    break

        self.history["epochs"] = list(range(it))

        if verbose:
            print()
            print("Exit at epoch: {}".format(it))

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

        return validation_err

    def sum_square_weights(self, size):
        """

        @param size: dimension of the batch
        It's used to regularization
        """
        sum = 0
        for layer in self.layers:
            sum += np.linalg.norm(layer.w) ** 2
        return sum / (2 * size)

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
        if show:
            plt.show()
        if path is not None:
            plt.savefig(path)

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
        if show:
            plt.show()
        if path is not None:
            plt.savefig(path)
