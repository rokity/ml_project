from neural_network import NeuralNetwork
from parser import *
from random_search import random_search
from utility import write_results


path_tr = 'data/monks/monks-2.train'
path_ts = 'data/monks/monks-2.test'
path_result_randomsearch = 'out/monks/monk2/randomsearch.csv'
path_err = 'out/monks/monk2/err_monk2'
path_acc = 'out/monks/monk2/acc_monk2'
path_result_bestmodel = 'out/monks/monk2/results.csv'

dim_in = 6
one_hot = 17
dim_hid = 3
dim_out = 1

f = 'sigmoid'
loss = 'lms'
acc = 'accuracy'
output_f = 'sigmoid'

parser = Monks_parser(path_tr, path_ts)

tr, _, ts = parser.parse(dim_in, dim_out, one_hot, None)

#tr.normalize_out_classification(0, -1)
#vl.normalize_out_classification(0, -1)
#ts.normalize_out_classification(0, -1)

vl = tr

if one_hot is not None:
    dim_in = one_hot

model = NeuralNetwork(loss=loss, acc=acc)
model.add_input_layer(dim_in, dim_hid, f, dim_hid)
model.add_output_layer(dim_out, output_f, dim_hid)

best_model = random_search(model, tr, vl, ts, max_evals=10, path_results=path_result_randomsearch, verbose=True)
err_tr, acc_tr = best_model.predict_dataset(tr)
err_ts, acc_ts = best_model.predict_dataset(ts)
acc_tr = acc_tr*100
acc_ts = acc_ts*100
errors = [err_tr, err_ts]
accuracy = [acc_tr, acc_ts]

res = {
    'Error': errors,
    'Accuracy': accuracy,
}

save = (path_err, path_acc, path_result_bestmodel)
write_results(res, best_model, save=None)
