from  utility import *
import numpy as np


PARAM_GRID = {
    'eta': list([0.001, 0.0001,0.01,0.1,0.00001]),
    'alpha': list([0.5,0.6,0.7,0.8,0.9]),
    'lambda': list(np.logspace(0, 5, 5).round(5)),
    'add_layer': list([True, False]),
    'hidden_nodes': list([ 10,20, 30, 40]),
    'hidden_nodes2': list([10,20, 30, 40,50]),
    'optimizer':list(['SGD','Adam','RMSprop']),
    'activation_function1':list(['tanh','sigmoid','relu','linear']),
    'activation_function2':list(['tanh','sigmoid','relu','linear']),
    'activation_function3':list(['tanh','sigmoid','relu','linear']),
    'init_weights':list(['XavierNormalInitialization','HeInitialization','RandomNormalInitialization']),
    'early_stopping':list(['GL','PQ',None]),
    'beta_1': list([0.9,0.99,0.999]),
    'beta_2': list([0.9,0.99,0.999]),
}
generate_hyperparameters_combination(PARAM_GRID,_random=False,path_params="params.csv")
