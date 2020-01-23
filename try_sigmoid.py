from parser import *
from functions_factory import FunctionsFactory
from neural_network import NeuralNetwork


#initialize parameters

path_tr = 'monks/monks-3.train'
path_ts = 'monks/monks-3.test'
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

parser = Monks_parser(path_tr, path_ts)
tr, vl, ts = parser.parse(dim_in, dim_out, one_hot, 0.3)

nn = NeuralNetwork(topology, f, loss, dim_hid, tr.size, 0.5, 0.8, 0.01)
err = nn.train(tr, vl, ts, 1e-2, 2000)
print("Validation error: {}\n".format(err))
nn.show_all_err()
nn.show_all_acc()
