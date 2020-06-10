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

DIR_DATA = "/home/fabsam/Documenti/masterDegree/ML/ml_project/data/monks/"
TR_FILE = 'monks-3.train'
TS_FILE = 'monks-3.test'
'''
path_tr = DIR_DATA + TR_FILE
path_ts = DIR_DATA + TS_FILE
path_result_randomsearch = 'out/monks/monk3/randomsearch.csv'
path_err = 'out/monks/monk1/err_monk3'
path_acc = 'out/monks/monk1/acc_monk3'
path_result_bestmodel = 'out/monks/monk3/results.csv'
'''
dim_in = 6
one_hot = 17
dim_out = 1
'''
parser = Monks_parser(path_tr, path_ts)

X_train, Y_train, X_test, Y_test = parser.parse(dim_in, dim_out, one_hot)

Y_train = utility.change_output_value(Y_train, 0, -1)
Y_test = utility.change_output_value(Y_test, 0, -1)
'''
import pandas as pd
train_data_df = pd.read_csv(DIR_DATA + TR_FILE, sep=" ", header=None, usecols=list(range(1, 8)))
test_data_df = pd.read_csv(DIR_DATA + TS_FILE, sep=" ", header=None, usecols=list(range(1, 8)))
Y = train_data_df[1].values
X = train_data_df[list(range(2, 8))].values
Y_test = test_data_df[1].values
X_test = test_data_df[list(range(2, 8))].values
X_train = X
Y_train = Y


Y_train = Y_train.reshape((Y_train.shape[0], 1))
Y_test = Y_test.reshape((Y_test.shape[0], 1))

Y_train = utility.change_output_value(Y_train, 0, -1)
Y_test = utility.change_output_value(Y_test, 0, -1)

from sklearn.preprocessing import OneHotEncoder

one_hot_enc = OneHotEncoder()
X_train = one_hot_enc.fit_transform(X_train).toarray()
X_test = one_hot_enc.transform(X_test).toarray()



dim_in = one_hot
dim_hid = 3

model = NeuralNetwork('mse', 'accuracy1-1')

model.add_layer(dim_hid, input_dim=dim_in, activation='sigmoid', kernel_initialization=RandomUniformInitialization())
model.add_output_layer(dim_out, activation='tanh', kernel_initialization=GlorotBengioInitialization(dim_hid))

model.compile(0.5, 0.9, 0.0)
model.fit(X_train, Y_train, 50, X_train.shape[0], ts=(X_test, Y_test), verbose=True, tol=1e-2)

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
