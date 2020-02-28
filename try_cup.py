from functions_factory import FunctionsFactory
from neural_network import NeuralNetwork
from parser import *

# initialize parameters
path_tr = 'cup/ML-CUP19-TR.csv'
path_ts = 'cup/ML-CUP19-TS.csv'
dim_in = 20
dim_hid = 15
dim_hid2 = 10
dim_out = 2
# activation functions
f = FunctionsFactory.build('sigmoid')
out_f = FunctionsFactory.build('linear')

# loss function
loss = FunctionsFactory.build('mse')

# accuracy function
acc = FunctionsFactory.build('mee')

topology = [dim_in, dim_hid, dim_hid2, dim_out]

parser = Cup_parser(path_tr)
tr, vl, ts = parser.parse(dim_in, dim_out)

print("tr size: {}".format(tr.size))
print("vl size: {}".format(vl.size))
print("ts size: {}".format(ts.size))

''' It can be used to feature scaling
avg = tr.features_scaling()
vl.features_scaling_avg(avg)
ts.features_scaling_avg(avg)
'''

nn = NeuralNetwork(topology, f, loss, acc, dim_hid+dim_hid2, tr.size, 0.1, 0.3, 0.01)
nn.set_out_actf(out_f)

err = nn.train(tr, vl, ts, 1e-2, 2000)

print("Validation error: {}\n".format(err))
#nn.save_trts_err('./out/1_all_err.png')
#nn.save_trts_acc('./out/1_all_acc.png')
nn.show_all_err()
nn.show_all_acc()
