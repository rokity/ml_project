from neural_network import NeuralNetwork
from parser import *
from random_search import random_search
from utility import write_results

# initialize parameters
path_tr = 'data/cup/ML-CUP19-TR.csv'
path_ts = 'data/cup/ML-CUP19-TS.csv'
path_result_randomsearch = 'out/cup/randomsearch.csv'
path_err = 'out/cup/mse_cup'
path_acc = 'out/cup/mee_cup'
path_result_bestmodel = 'out/cup/results.csv'

# activation functions
f = 'sigmoid'
out_f = 'linear'

# loss function
loss = 'mse'

# accuracy function
acc = 'mee'

dim_in = 20
dim_hid = 15
dim_hid2 = 10
dim_out = 2
fan_in = dim_hid + dim_hid2

parser = Cup_parser(path_tr)
tr, vl, ts = parser.parse(dim_in, dim_out)


''' It can be used to feature scaling
avg = tr.features_scaling()
vl.features_scaling_avg(avg)
ts.features_scaling_avg(avg)
'''

model = NeuralNetwork(loss=loss, acc=acc)
model.add_input_layer(dim_in, dim_hid, f, fan_in)
model.add_hidden_layer(dim_hid2, f, fan_in)
model.add_output_layer(dim_out, out_f, fan_in)

best_model = random_search(model, tr, vl, ts, max_evals=16, path_results=path_result_randomsearch, verbose=True, n_threads=8)

err_tr, acc_tr = best_model.predict_dataset(tr)
err_ts, acc_ts = best_model.predict_dataset(ts)
errors = [err_tr, err_ts]
accuracy = [acc_tr, acc_ts]

res = {
    'mse': errors,
    'mee': accuracy,
}

save = (path_err, path_acc, path_result_bestmodel)
write_results(res, best_model, save=None, all=True)
