from parser import Parser
from functions_factory import FunctionsFactory
from neural_network import NeuralNetwork
import numpy as np
import time

#initialize parameters

path_tr = 'monks/monks-2.train'
path_ts = 'monks/monks-2.test'
dim_in = 6
one_hot = 17
dim_hid = 4
dim_out = 1
f = FunctionsFactory.build('sigmoid')
loss = FunctionsFactory.build('lms')

if one_hot is None:
    topology = [dim_in, dim_hid, dim_out]
else:
    topology = [one_hot, dim_hid, dim_out]

tr, vl, ts = Parser.parse(path_tr, path_ts, dim_in, dim_out, one_hot, None)
nn = NeuralNetwork(topology, f, loss, dim_hid, tr.size, 0.5, 0.8, 0.01)
err = nn.train(tr, tr, ts, 0.02, 2000)
print("Final error: {}\n".format(err))
#nn.save_trts_err('./out/1_trts_err.png')
#nn.save_trts_acc('./out/1_trts_acc.png')
nn.show_trts_err()
nn.show_trts_acc()
