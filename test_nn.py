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


DIR_DATA = "/home/fabsam/Documenti/masterDegree/ML/ml_project/data/monks/"
TR_FILE = 'monks-3.train'
TS_FILE = 'monks-3.test'


path_tr = DIR_DATA + TR_FILE
path_ts = DIR_DATA + TS_FILE

dim_in = 6
one_hot = 17
dim_out = 1

parser = Monks_parser(path_tr, path_ts)

X_train, Y_train, X_test, Y_test = parser.parse(dim_in, dim_out, one_hot)

Y_train = utility.change_output_value(Y_train, 0, -1)
Y_test = utility.change_output_value(Y_test, 0, -1)

dim_in = one_hot
dim_hid = 3

model = NeuralNetwork('mse', 'accuracy')

model.add_layer(dim_hid, input_dim=dim_in, activation='sigmoid', kernel_initialization=RandomNormalInitialization())
model.add_output_layer(dim_out, activation='tanh', kernel_initialization=GlorotBengioInitialization(dim_hid))

model.compile(0.3, 0.8, 0.001)
model.fit(X_train, Y_train, 500, X_train.shape[0], ts=(X_test, Y_test), verbose=True, tol=1e-2)

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
