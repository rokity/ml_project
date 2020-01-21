from parser import Parser
from functions_factory import FunctionsFactory
from neural_network import NeuralNetwork
import numpy as np
import time

path_tr = '../problems/monkas/monks-3.train'
path_ts = '../problems/monkas/monks-3.test'
dim_in = 6
one_hot = 17
dim_hid = 4
dim_out = 1
f = FunctionsFactory.build('tanh')
loss = FunctionsFactory.build('lms')
if one_hot is None:
    topology = [dim_in, dim_hid, dim_out]
else:
    topology = [one_hot, dim_hid, dim_out]
tr = Parser.parse(path_tr, dim_in, dim_out, one_hot)
ts = Parser.parse(path_ts, dim_in, dim_out, one_hot)
tr.normalize_out()
ts.normalize_out()
nn = NeuralNetwork(topology, f, loss, dim_hid, tr.size, 0.1, 0.5)
nn.train(tr, ts, 1e-2, 2000)
nn.show_trts_err()
nn.show_trts_acc()
