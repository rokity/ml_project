from neural_network import NeuralNetwork
from optimizers import *
from parser import Cup_parser
import time
import os
from model_selection import *

DIR_CUP = './data/cup/'
PATH_TR = 'ML-CUP19-TR.csv'
PATH_TS = 'ML-CUP19-TS.csv'
path_result_randomsearch='./out/cup/search.csv'
'''
path_result_model = './out/cup/grid_search_model.csv'
path_plot_loss = os.getcwd()+'/out/cup/grid_search_plot_loss.png'
path_plot_metric = os.getcwd()+'/out/cup/grid_search_plot_metric.png'
'''

INPUT_DIM = 20
OUTPUT_DIM = 2
PERC_TEST = 0.25
LOSS = 'mse'
METRIC = 'mee'
K = 4


PARAM_SEARCH = {
    'eta': list([0.001, 0.0001,0.01]),
    #'momentum': list([0.7, 0.8, 0.9]),
    'lambda': list([0.0002,0.0004,0.0005,0.0009]),
    'add_layer': list([True, False]),
    'hidden_nodes': list([20, 30, 40]),
    'hidden_nodes2': list([20, 30, 40]),
    'optimizer': list(['Adam', 'RMSprop']),
    'activation_function1': list(['tanh', 'sigmoid']),
    'activation_function2': list(['tanh', 'sigmoid']),
    'init_weights': list(['XavierNormalInitialization','HeInitialization',  'RandomNormalInitialization']),
    'batch_size': list([8, 16, 32, 64]),
    'beta_1': list([0.9, 0.99, 0.999]),
    'beta_2': list([0.9, 0.99, 0.999]),
}


def create_model(hyperparams):
    _module_kernelinit = __import__('kernel_initialization')

    lr = hyperparams['eta']
    #mom = hyperparams['momentum']
    l2 = hyperparams['lambda']
    beta_1 = hyperparams['beta_1']
    beta_2 = hyperparams['beta_2']
    act_fun_1 = hyperparams['activation_function1']
    hidden_nodes1 = hyperparams['hidden_nodes']
    kernel_init = getattr(_module_kernelinit, hyperparams['init_weights'])
    optimizer_str = hyperparams['optimizer']

    model = NeuralNetwork(LOSS, METRIC)
    model.add_layer(hidden_nodes1, input_dim=INPUT_DIM, activation=act_fun_1, kernel_initialization=kernel_init())
    if hyperparams['add_layer']:
        act_fun_2 = hyperparams['activation_function2']
        hidden_nodes2 = hyperparams['hidden_nodes2']
        model.add_layer(hidden_nodes2, activation=act_fun_2, kernel_initialization=kernel_init())
    model.add_layer(OUTPUT_DIM, activation='linear', kernel_initialization=kernel_init())
    if optimizer_str == "SGD":
        optimizer = SGD(lr=lr, mom=mom, l2=l2)
    elif optimizer_str == "RMSprop":
        optimizer = RMSprop(lr=lr, moving_average=beta_1, l2=l2)
    elif optimizer_str == "Adam":
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, l2=l2)
    else:
        raise ValueError("You have to choose an existing optimizer")

    model.compile(optimizer=optimizer)
    return model


start_time = time.time()
set_style_plot()

parser = Cup_parser(DIR_CUP + PATH_TR)
data, targets = parser.parse(INPUT_DIM, OUTPUT_DIM)

X_test, Y_test, folds_X, folds_Y = train_val_test_split_k_fold(data, targets, test_size=PERC_TEST, shuffle=True, k_fold=K)

<<<<<<< HEAD
model = random_search(
=======
model=random_search(
>>>>>>> 59f162b2eec9a044e93427d6b43a23c80734f3d0
    create_model=create_model, tr=(folds_X, folds_Y), k_fold=K, epochs=500, max_evals=50,
    batch_size=None, param_grid=PARAM_SEARCH, monitor_value='val_mee',
    vl=None, ts=(X_test, Y_test), path_results=path_result_randomsearch,n_threads=50, verbose=True, shuffle=True
)

print(time.time() - start_time, "seconds")

'''
write_results(res, model=None, save_result=path_result_model, test=False, validation=True)
model.plot_loss(val=True, test=True, show=False, path=path_plot_loss)
model.plot_metric(val=True, test=True, show=False, path=path_plot_metric)
'''