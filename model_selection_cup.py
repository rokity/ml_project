from neural_network import NeuralNetwork
from optimizers import *
from kernel_initialization import *
from functions_factory import *
from parser import Cup_parser
from utility import set_style_plot, train_test_split
from random_search import random_search

DIR_CUP = './data/cup/'
PATH_TR = 'ML-CUP19-TR.csv'
PATH_TS = 'ML-CUP19-TS.csv'

INPUT_DIM = 20
OUTPUT_DIM = 2
PERC_TEST = 0.25
LOSS = 'mse'
METRIC = 'mee'
K = 4

PARAM_SEARCH = {
    'lr': list([0.1, 0.01, 0.001, 0.0001, 0.00001]),
    'momentum': list([0.5, 0.6, 0.7, 0.8, 0.9]),
    'l2': list(np.logspace(0, 5, 5).round(5)),
    'batch_size': list([1, 4, 8, 16, 32, 64]),
    'add_layer': list([True, False]),
    'hidden_nodes': list([10, 20, 30, 40]),
    'hidden_nodes2': list([10, 20, 30, 40, 50]),
    'optimizer': list(['SGD', 'Adam', 'RMSprop']),
    'activation_function1': list(['tanh', 'sigmoid']),
    'activation_function2': list(['tanh', 'sigmoid']),
    'init_weights': list(['XavierNormalInitialization', 'HeInitialization', 'RandomNormalInitialization']),
    'beta_1': list([0.9, 0.99, 0.999]),
    'beta_2': list([0.9, 0.99, 0.999]),
}


def create_model(hyperparams):
    _module_kernelinit = __import__('kernel_initialization')

    lr = hyperparams['l2']
    mom = hyperparams['momentum']
    l2 = hyperparams['l2']
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


set_style_plot()

parser = Cup_parser(DIR_CUP + PATH_TR)
data, targets = parser.parse(INPUT_DIM, OUTPUT_DIM)
X_train, Y_train, X_test, Y_test = train_test_split(data, targets, test_size=PERC_TEST, shuffle=True)

random_search(
    create_model=create_model, tr=(X_train, Y_train), k_fold=K, epochs=500,
    batch_size=None, param_grid=PARAM_SEARCH, monitor_value='val_mee',
    max_evals=10, verbose=True, shuffle=True
)




