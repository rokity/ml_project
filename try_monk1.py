from functions_factory import FunctionsFactory
from neural_network import NeuralNetwork
from parser import *

# initialize parameters
path_tr = 'monks/monks-1.train'
path_ts = 'monks/monks-1.test'
dim_in = 6
one_hot = 17
dim_hid = 4
dim_out = 1
f = FunctionsFactory.build('sigmoid')
last_f = FunctionsFactory.build('tanh')
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

nn = NeuralNetwork(topology, f, loss, acc, dim_hid, tr.size, 0.1, 0.1, 0.01)
nn.set_out_actf(last_f)
err = nn.train(tr, tr, ts, 1e-2, 3000)

print("Validation error: {}\n".format(err))
#nn.save_trts_err('./monk_out/1_all_err.png')
#nn.save_trts_acc('./monk_out/1_all_acc.png')
nn.show_trts_err()
nn.show_trts_acc()
