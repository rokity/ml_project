from neural_network import NeuralNetwork
from parser import *
from utility import set_style_plot
from kernel_initialization import *
from utility import write_results
from random_search import random_search
from utility import write_results


set_style_plot()

path_tr = 'data/monks/monks-1.train'
path_ts = 'data/monks/monks-1.test'
path_result_randomsearch = 'out/monks/monk1/randomsearch.csv'
path_err = 'out/monks/monk1/err_monk1'
path_acc = 'out/monks/monk1/acc_monk1'
path_result_bestmodel = 'out/monks/monk1/results.csv'

dim_in = 6
one_hot = 17
dim_hid = 4
dim_out = 1

parser = Monks_parser(path_tr, path_ts)

tr, _, ts = parser.parse(dim_in, dim_out, one_hot, perc_val=None)

tr.normalize_out_classification(0, -1)
#vl.normalize_out_classification(0, -1)
ts.normalize_out_classification(0, -1)


if one_hot is not None:
    dim_in = one_hot

model = NeuralNetwork(loss='lms', metric='accuracy1-1')
model.add_input_layer(dim_in, dim_hid, 'sigmoid', RandomUniformInitialization())
model.add_output_layer(dim_out, 'tanh', GlorotBengioInitialization(dim_hid))


eta = 0.2
mom = 0.5
lam = 0.0
model.compile(lr=eta, momentum=mom)
err = model.fit(tr, batch_size=tr.size, epochs=2000, vl=None, ts=ts, verbose=True, tol=1e-2)

model.plot_loss(val=False, test=True, show=True)
model.plot_metric(val=False, test=True, show=True)

'''

PARAM_GRID = {
    'alpha': list(np.linspace(0.1, 1, 10).round(2)),
    'eta': list(np.linspace(0.1, 1, 10).round(2)),
    'lambda': list([0])
}

best_model = random_search(model, tr, vl, ts, max_evals=30, param_grid=PARAM_GRID, path_results=path_result_randomsearch, verbose=True)

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
'''
