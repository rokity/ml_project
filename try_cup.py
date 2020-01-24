from functions_factory import FunctionsFactory
from neural_network import NeuralNetwork
from parser import *

# initialize parameters
path_tr = 'cup/ML-CUP19-TR.csv'
path_ts = 'cup/ML-CUP19-TS.csv'
dim_in = 20
dim_hid = 4
dim_out = 2
f = FunctionsFactory.build('sigmoid')
loss = FunctionsFactory.build('lms')
acc = FunctionsFactory.build('accuracy')

topology = [dim_in, dim_hid, dim_out]

parser = Cup_parser(path_tr)
tr, vl, ts = parser.parse(dim_in, dim_out)
'''
nn = NeuralNetwork(topology, f, loss, acc, dim_hid, tr.size, 0.5, 0.8, 0.01)
err = nn.train(tr, vl, ts, 1e-2, 2000)

print("Validation error: {}\n".format(err))
#nn.save_trts_err('./out/1_all_err.png')
#nn.save_trts_acc('./out/1_all_acc.png')
nn.show_all_err()
nn.show_all_acc()
'''