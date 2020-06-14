from neural_network import NeuralNetwork
from parser import Monks_parser
from utility import set_style_plot
from utility import *
from kernel_initialization import *

set_style_plot()

DIR_DATA = "/home/fabsam/Documenti/masterDegree/ML/ml_project/data/monks/"
TR_FILE = 'monks-1.train'
TS_FILE = 'monks-1.test'


path_tr = DIR_DATA + TR_FILE
path_ts = DIR_DATA + TS_FILE
path_loss = 'out/monks/monk1/err_monk1.png'
path_acc = 'out/monks/monk1/acc_monk1.png'
path_result_model = 'out/monks/monk1/results.csv'

dim_in = 6
one_hot = 17
dim_out = 1

parser = Monks_parser(path_tr, path_ts)

X_train, Y_train, X_test, Y_test = parser.parse(dim_in, dim_out, one_hot)


Y_train = change_output_value(Y_train, 0, -1)
Y_test = change_output_value(Y_test, 0, -1)


dim_in = one_hot
dim_hid = 4
n_attemps = 5

res_losses = np.zeros(5)
res_metrics = np.zeros(5)

res_losses_ts = np.zeros(n_attemps)
res_metrics_ts = np.zeros(n_attemps)

for i in range(n_attemps):
    model = NeuralNetwork('mse', 'accuracy1-1')

    model.add_layer(dim_hid, input_dim=dim_in, activation='sigmoid', kernel_initialization=RandomUniformInitialization(-1, 1))
    model.add_output_layer(dim_out, activation='tanh', kernel_initialization=RandomNormalInitialization(0, 0.5))

    model.compile(0.8, 0.7, 0.0)
    history = model.fit(
        X_train, Y_train, 500, X_train.shape[0], ts=(X_test, Y_test),
        verbose=True, tol=1e-3
    )
    if i == 0:
        write_results(res=None, model=model, save_plot_loss=path_loss, save_plot_metric=path_acc, test=True)
    res_losses[i] = history['mse'][-1]
    res_losses_ts[i] = history['test_mse'][-1]
    res_metrics[i] = history['accuracy'][-1]*100
    res_metrics_ts[i] = history['test_accuracy'][-1]*100
    print("tr acc ", res_metrics[i])
    print("ts acc ", res_metrics_ts[i])

errors = [res_losses.mean(), res_losses_ts.mean()]
accuracy = [res_metrics.mean(), res_metrics_ts.mean()]

res = {
    'Error': errors,
    'Accuracy': accuracy,
}

print(res)
write_results(res, model=None, save_result=path_result_model, test=True)
