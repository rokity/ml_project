from neural_network import NeuralNetwork
from parser import Monks_parser
from utility import set_style_plot
import utility
from kernel_initialization import *
from random_search import random_search
from utility import write_results
from utility import write_results

set_style_plot()

PARAM_GRID = {
    'alpha': list(np.linspace(0.1, 1, 10).round(2)),
    'eta': list(np.linspace(0.1, 1, 10).round(2)),
    'hidden_nodes': list(np.linspace(2, 4, 3, dtype=np.uint8))
}


def create_model(hyperparams):
    lr = hyperparams['eta']
    mom = hyperparams['alpha']
    dim_hid = int(hyperparams['hidden_nodes'])

    model = NeuralNetwork(loss='mse', metric='accuracy1-1')
    model.add_layer(dim_hid, input_dim=dim_in, activation='sigmoid', kernel_initialization=RandomUniformInitialization())
    model.add_output_layer(dim_out, activation='tanh', kernel_initialization=GlorotBengioInitialization(dim_hid))

    model.compile(lr, mom)

    return model


path_tr = 'data/monks/monks-2.train'
path_ts = 'data/monks/monks-2.test'
path_result_randomsearch = 'out/monks/monk2/randomsearch.csv'
path_err = 'out/monks/monk1/err_monk2'
path_acc = 'out/monks/monk1/acc_monk2'
path_result_bestmodel = 'out/monks/monk2/results.csv'

dim_in = 6
one_hot = 17
dim_out = 1

parser = Monks_parser(path_tr, path_ts)

X_train, Y_train, X_test, Y_test = parser.parse(dim_in, dim_out, one_hot)

Y_train = utility.change_output_value(Y_train, 0, -1)
Y_test = utility.change_output_value(Y_test, 0, -1)

dim_in = one_hot

model = random_search(
    create_model,
    (X_train, Y_train),
    (X_train, Y_train),
    500,
    X_train.shape[0],
    param_grid=PARAM_GRID,
    monitor_value='mse',
    ts=(X_test, Y_test),
    max_evals=50,
    n_threads=1,
    #path_results=path_result_randomsearch,
    tol=1e-3,
    verbose=True
)

model.plot_loss(val=False, test=True, show=True)
model.plot_metric(val=False, test=True, show=True)

err_tr, acc_tr = model.evaluate(X_train, Y_train)
err_ts, acc_ts = model.evaluate(X_test, Y_test)
acc_tr = acc_tr*100
acc_ts = acc_ts*100
errors = [err_tr, err_ts]
accuracy = [acc_tr, acc_ts]

res = {
    'Error': errors,
    'Accuracy': accuracy,
}

print(res)
#save = (path_err, path_acc, path_result_bestmodel)
#write_results(res, best_model, save=None)
