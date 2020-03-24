from layer import *
from dataset import Dataset
from functions_factory import FunctionsFactory
import matplotlib.pyplot as plt
import sys


class NeuralNetwork:
    def __init__(self, loss, acc):
        self.layers = []
        self.loss = FunctionsFactory.build(loss)
        self.acc = FunctionsFactory.build(acc)
        self.l_tr_err = []
        self.l_vl_err = []
        self.l_ts_err = []
        self.l_tr_acc = []
        self.l_vl_acc = []
        self.l_ts_acc = []
        self.l_it = []

    def get_vl_error(self):
        return self.l_vl_err[-1]

    def compile(self, eta, alpha, lam, batch_size=1):
        self.eta = eta
        self.alpha = alpha
        self.lam = lam
        self.batch_size = batch_size
        for i in range(len(self.layers)):
            self.layers[i].compile()

    def add_input_layer(self, dim_in, dim_out, f_act, fan_in=1, label='Input layer'):
        f_act = FunctionsFactory.build(f_act)
        layer = Layer(dim_in, dim_out, f_act, self.loss, fan_in, label)
        self.layers.append(layer)

    def add_hidden_layer(self, dim_out, f_act, fan_in=1, label='Hidden layer'):
        f_act = FunctionsFactory.build(f_act)
        layer = Layer(self.layers[-1].dim_out, dim_out, f_act, self.loss, fan_in, label)
        self.layers.append(layer)

    def add_output_layer(self, dim_out, f_act, fan_in=1, label='Input layer'):
        f_act = FunctionsFactory.build(f_act)
        layer = OutputLayer(self.layers[-1].dim_out, dim_out, f_act, self.loss, fan_in, label)
        self.layers.append(layer)

    def predict(self, x: np.ndarray):
        return self.__feedforward(x)

    def predict_dataset(self, dataset: Dataset):
        return self.__feedforward_dataset(dataset)

    def __feedforward(self, x: np.ndarray):
        for layer in self.layers:
            x = layer.feedforward(x)
        return x

    def __feedforward_dataset(self, dataset: Dataset):
        err = 0
        acc = 0
        for i in range(dataset.size):
            x, d = dataset.get_data(i)
            x = x.reshape(x.shape[0], 1)
            y = self.__feedforward(x.T)
            err += self.loss.compute_fun(d, y)
            acc += self.acc.compute_fun(d, y)
        return float(err) / dataset.size, acc / dataset.size

    def backpropagation(self, d):
        loc_grad = None
        for layer in reversed(self.layers):
            loc_grad = layer.backpropagation(loc_grad, d)

    def update_weights(self):
        for layer in self.layers:
            layer.update_weights(self.eta, self.alpha, self.lam, self.batch_size)

    def train(self, epochs, tr, vl, ts=None, verbose=False):
        it = 0
        curr_i = 0
        training_err = 0
        validation_err = 0
        min_tr_err = sys.float_info.max
        min_vl_err = sys.float_info.max

        while it < epochs:

            # train
            for _ in range(self.batch_size):
                x, d = tr.get_data(curr_i)
                x = x.reshape(x.shape[0], 1)
                d = d.reshape(d.shape[0], 1)
                y = self.__feedforward(x.T)
                self.backpropagation(d.T)
                curr_i = (curr_i + 1) % tr.size

            # compute error in training set
            err, acc = self.__feedforward_dataset(tr)
            err = err
            self.l_tr_err.append(err)
            self.l_tr_acc.append(acc)
            tot_weights = self.sum_square_weights(tr.size)

            training_err = err + (self.lam * tot_weights)

            min_tr_err = min(min_tr_err, training_err)

            # compute error in validation set
            err, acc = self.__feedforward_dataset(vl)
            self.l_vl_err.append(err)
            self.l_vl_acc.append(acc)

            validation_err = err

            # generalization loss
            gl = 100 * ((validation_err / min_vl_err) - 1)

            min_vl_err = min(min_vl_err, validation_err)

            # compute error in test set
            if ts is not None:
                err, acc = self.__feedforward_dataset(ts)
                self.l_ts_err.append(err)
                self.l_ts_acc.append(acc)

            self.l_it.append(it)

            self.update_weights()

            if verbose:
                print(
                    "It {}:\t tr_err: {},\t vl_err: {},\t gl: {}".format(it, training_err, validation_err, gl),
                    end='\r'
                )

            if gl > 0.2 and training_err - min_tr_err > 0:
                break
            it += 1

        if verbose:
            print("Exit at epoch: {}".format(it))

            print("{} training set: {}".format(self.loss.name, self.l_tr_err[-1]))
            print("{} validation set: {}".format(self.loss.name, self.l_vl_err[-1]))
            print("{} test set: {}".format(self.loss.name, self.l_ts_err[-1]))
            if self.acc.name == 'accuracy':
                print("% accuracy training set: {}".format(self.l_tr_acc[-1] * 100))
                print("% accuracy validation set: {}".format(self.l_vl_acc[-1] * 100))
                print("% accuracy test set: {}".format(self.l_ts_acc[-1] * 100))
            else:
                print("{} training set: {}".format(self.acc.name, self.l_tr_acc[-1]))
                print("{} validation set: {}".format(self.acc.name, self.l_vl_acc[-1]))
                print("{} accuracy test set: {}".format(self.acc.name, self.l_ts_acc[-1]))

        return validation_err

    def sum_square_weights(self, size):
        sum = 0
        for layer in self.layers:
            sum += np.linalg.norm(layer.w) ** 2
        return sum / (2 * size)

    def plot_tr_err(self):
        plt.plot(self.l_it, self.l_tr_err, 'r', label='TR error')

    def plot_vl_err(self):
        plt.plot(self.l_it, self.l_vl_err, 'g:', label='VL error')

    def plot_ts_err(self):
        plt.plot(self.l_it, self.l_ts_err, 'b-.', label='TS error')

    def plot_tr_acc(self):
        plt.plot(self.l_it, self.l_tr_acc, 'r', label='TR accuracy')

    def plot_vl_acc(self):
        plt.plot(self.l_it, self.l_vl_acc, 'g:', label='VL accuracy')

    def plot_ts_acc(self):
        plt.plot(self.l_it, self.l_ts_acc, 'b-.', label='TS accuracy')

    def show_trts_err(self):
        self.plot_tr_err()
        self.plot_ts_err()
        plt.xlabel('epochs')
        plt.ylabel(self.loss.name)
        plt.legend()
        plt.show()

    def show_trvl_err(self):
        self.plot_tr_err()
        self.plot_vl_err()
        plt.xlabel('epochs')
        plt.ylabel(self.loss.name)
        plt.legend()
        plt.show()

    def show_all_err(self):
        self.plot_tr_err()
        self.plot_vl_err()
        self.plot_ts_err()
        plt.xlabel('epochs')
        plt.ylabel(self.loss.name)
        plt.legend()
        plt.show()

    def show_trts_acc(self):
        self.plot_tr_acc()
        self.plot_ts_acc()
        plt.xlabel('epochs')
        plt.ylabel(self.acc.name)
        plt.legend()
        plt.show()

    def show_trvl_acc(self):
        self.plot_tr_err()
        self.plot_vl_err()
        plt.xlabel('epochs')
        plt.ylabel(self.acc.name)
        plt.legend()
        plt.show()

    def show_all_acc(self):
        self.plot_tr_acc()
        self.plot_vl_acc()
        self.plot_ts_acc()
        plt.xlabel('epochs')
        plt.ylabel(self.acc.name)
        plt.legend()
        plt.show()

    def save_trts_err(self, path):
        plt.figure()
        self.plot_tr_err()
        self.plot_ts_err()
        plt.xlabel('epochs')
        plt.ylabel(self.loss.name)
        plt.legend()
        plt.savefig(path)

    def save_trts_acc(self, path):
        plt.figure()
        self.plot_tr_acc()
        self.plot_ts_acc()
        plt.xlabel('epochs')
        plt.ylabel(self.acc.name)
        plt.legend()
        plt.savefig(path)

    def save_all_err(self, path):
        plt.figure()
        self.plot_tr_err()
        self.plot_vl_err()
        self.plot_ts_err()
        plt.xlabel('epochs')
        plt.ylabel(self.loss.name)
        plt.legend()
        plt.savefig(path)

    def save_all_acc(self, path):
        plt.figure()
        self.plot_tr_acc()
        self.plot_vl_acc()
        self.plot_ts_acc()
        plt.xlabel('epochs')
        plt.ylabel(self.acc.name)
        plt.legend()
        plt.savefig(path)

