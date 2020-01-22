from parser import Parser
from functions_factory import FunctionsFactory
from neural_network import NeuralNetwork
import numpy as np
import time

path_tr = '../problems/monkas/monks-1.train'
path_ts = '../problems/monkas/monks-1.test'
dim_in = 6
one_hot = 17
dim_hid = 3
dim_out = 1
f = FunctionsFactory.build('tanh')
loss = FunctionsFactory.build('lms')
if one_hot is None:
    topology = [dim_in, dim_hid, dim_out]
else:
    topology = [one_hot, dim_hid, dim_out]
tr, vl, ts = Parser.parse(path_tr, path_ts, dim_in, dim_out, one_hot, None)
tr.normalize_out()
vl.normalize_out()
ts.normalize_out()
nn = NeuralNetwork(topology, f, loss, dim_hid, tr.size, 0.5, 0.7, 0)
nn.train(tr, vl, ts, 1e-3, 2000)
nn.show_trts_err()
nn.show_trts_acc()
