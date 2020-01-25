from layer import *
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, topology, f_act, loss, fan_in, batch_size=1, eta=0.5, alpha=0, lam=0):
        self.layers = []
        self.loss = loss
        self.batch_size = batch_size
        self.eta = eta
        self.alpha = alpha
        self.lam = lam
        self.__init_layers(topology, f_act, loss, fan_in)
        self.l_tr_err = []
        self.l_ts_err = []
        self.l_tr_acc = []
        self.l_ts_acc = []
        self.l_it = []

    def __init_layers(self, topology, f_act, loss, fan_in):
        for i in range(len(topology)-1):
            if i == len(topology)-2:
                layer = OutputLayer(topology[i], topology[i + 1], f_act, loss, fan_in, 'Layer ' + str(i))
            else:
                layer = Layer(topology[i], topology[i+1], f_act, loss, fan_in, 'Layer ' + str(i))
            # layer.print_info()
            self.layers.append(layer)

    def feedforward(self, x):
        for layer in self.layers:
            x = layer.feedforward(x)
        return x

    def backpropagation(self, d):
        loc_grad = None
        for layer in reversed(self.layers):
            loc_grad = layer.backpropagation(loc_grad, d)

    def update_weights(self):
        for layer in self.layers:
            layer.update_weights(self.eta, self.alpha, self.lam, self.batch_size)

    def __acc(self, d, y):
        if np.abs(d - y) < 0.5:
            return 1
        else:
            return 0

    def train(self, tr, vl, ts, epsilon, epochs):
        it = 0
        curr_i = 0
        final_err = 0
        while it < epochs:
            tr_err = 0
            ts_err = 0
            tr_acc = 0
            ts_acc = 0

            # train
            for _ in range(self.batch_size):
                x, d = tr.get_data(curr_i)
                x = x.reshape(x.shape[0], 1)
                y = self.feedforward(x.T)
                self.backpropagation(d)
                curr_i = (curr_i + 1) % tr.size

            '''
            # compute error in trainig set
            for i in range(tr.size):
                x, d = tr.get_data(i)
                x = x.reshape(x.shape[0], 1)
                y = self.feedforward(x.T)
                tr_err += self.loss.compute_fun(d, y)
                tr_acc += self.__acc(d, y)
            '''

            #TODO: add compute validation error

            #compute error in validation set
            for i in range(vl.size):
                x, d = vl.get_data(i)
                x = x.reshape(x.shape[0], 1)
                y = self.feedforward(x.T)
                tr_err += self.loss.compute_fun(d, y)
                tr_acc += self.__acc(d, y)

            tot_weights = self.sum_square_weights(tr.size)

            tr_err = tr_err / tr.size + self.lam*tot_weights
            final_err = tr_err
            self.l_tr_err.append(tr_err.item())
            tr_acc = tr_acc / tr.size
            self.l_tr_acc.append(tr_acc)

            if ts != None:

                #compute error in test set
                for i in range(ts.size):
                    x, d = ts.get_data(i)
                    x = x.reshape(x.shape[0], 1)
                    y = self.feedforward(x.T)
                    ts_err += self.loss.compute_fun(d, y)
                    ts_acc += self.__acc(d, y)

                ts_err = ts_err / ts.size
                self.l_ts_err.append(ts_err.item())
                ts_acc = ts_acc / ts.size
                self.l_ts_acc.append(ts_acc)

            self.l_it.append(it)

            self.update_weights()

            # print("Error train it {}: {}".format(it, tr_err))
            if tr_err < epsilon:
                break
            it += 1
        if it == epochs:
            print("End epochs")
        return final_err

    def sum_square_weights(self, size):
        sum = 0
        for layer in self.layers:
            sum += np.linalg.norm(layer.w)**2
        return sum / (2*size)

    '''
    def show_tr_err(self):
        plt.plot(self.l_it, self.l_tr_err)
        plt.show()

    def save_tr_err(self):
        plt.plot(self.l_it, self.l_tr_err)
        plt.xlabel('epochs')
        plt.ylabel(self.loss.name)
        plt.savefig('./out/tr_err.png')

    def save_ts_err(self):
        plt.plot(self.l_it, self.l_ts_err)
        plt.xlabel('epochs')
        plt.ylabel(self.loss.name)
        plt.savefig('./out/ts_err.png')
    '''

    def show_trts_err(self):
        plt.plot(self.l_it, self.l_tr_err, 'r')
        plt.plot(self.l_it, self.l_ts_err, 'b')
        plt.xlabel('epochs')
        plt.ylabel(self.loss.name)
        plt.show()

    def save_trts_err(self, path):
        plt.figure()
        plt.plot(self.l_it, self.l_tr_err, 'r')
        plt.plot(self.l_it, self.l_ts_err, 'b')
        plt.xlabel('epochs')
        plt.ylabel(self.loss.name)
        plt.savefig(path)

    def show_trts_acc(self):
        plt.plot(self.l_it, self.l_tr_acc, 'r')
        plt.plot(self.l_it, self.l_ts_acc, 'b')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.show()

    def save_trts_acc(self, path):
        plt.figure()
        plt.plot(self.l_it, self.l_tr_acc, 'r')
        plt.plot(self.l_it, self.l_ts_acc, 'b')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.savefig(path)



