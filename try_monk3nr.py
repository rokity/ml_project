from functions_factory import FunctionsFactory
from neural_network import NeuralNetwork
from parser import *
import matplotlib.pyplot as plt

# initialize parameters
path_tr = 'monks/monks-3.train'
path_ts = 'monks/monks-3.test'
dim_in = 6
one_hot = 17
dim_hid = 4
dim_out = 1
f = FunctionsFactory.build('sigmoid')
act_f = FunctionsFactory.build('tanh')

loss = FunctionsFactory.build('lms')

acc = FunctionsFactory.build('accuracy')

if one_hot is None:
    topology = [dim_in, dim_hid, dim_out]
else:
    topology = [one_hot, dim_hid, dim_out]

parser = Monks_parser(path_tr, path_ts)
tr, _, ts = parser.parse(dim_in, dim_out, one_hot, None)

tr.normalize_out()
#vl.normalize_out()
ts.normalize_out()

nn = NeuralNetwork(topology, f, loss, acc, dim_hid, tr.size, 0.3, 0.5, 0.0)
nn.set_out_actf(act_f)

err = nn.train(tr, tr, ts, 1e-2, 4000)
print("Validation error: {}\n".format(err))

#nn.save_trts_err('./monk_out/3nr_all_err.png')
#nn.save_trts_acc('./monk_out/3nr_all_acc.png')

nn.show_trts_err()
nn.show_trts_acc()