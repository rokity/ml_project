from parser import Cup_parser
from utility import *
from neural_network import NeuralNetwork
from kernel_initialization import *
from random_search import random_search
from optimizers import *
import time


DIR_CUP = './data/cup/'
PATH_TR = 'ML-CUP19-TR.csv'
PATH_TS = 'ML-CUP19-TS.csv'
INPUT_DIM = 20
OUTPUT_DIM = 2
path_result_randomsearch='./out/cup/randomsearch.csv'
path_result_model='./out/cup/random_search_model.csv'

PERC_VL = 0.25
PERC_TS = 0.25

set_style_plot()


PARAM_GRID = {
    'eta': list([0.001, 0.0001,0.01]),
    'alpha': list([0.7,0.8,0.9]),
    'lambda': list([0.0002,0.0003,0.0004,0.0005,0.0009]),
    'rho':list([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
    'add_layer': list([True, False]),
    'hidden_nodes': list([ 20, 30, 40]),
    'hidden_nodes2': list([20, 30, 40]),
    'optimizer':list(['SGD','Adam','RMSprop']),
    'activation_function1':list(['tanh','sigmoid']),
    'activation_function2':list(['tanh','sigmoid','linear']),
    'init_weights':list(['XavierNormalInitialization','HeInitialization','RandomNormalInitialization']),
    #'early_stopping':list(['GL','PQ',None]),
    'batch_size':list([1, 2, 4, 8, 16, 32, 64]),
    'beta_1': list([0.9,0.99,0.999]),
    'beta_2': list([0.9,0.99,0.999]),
    #'k_fold': list([1,4,5,10,20]),
}

for k, v in PARAM_GRID.items():
    print(k, ",  ", v)


def create_model(hyperparams):
    lr = hyperparams['eta']
    l2 = hyperparams['lambda']

    dim_hid = int(hyperparams['hidden_nodes'])
    add_layer=hyperparams['add_layer']
    activation_function1=hyperparams['activation_function1']
    activation_function2=hyperparams['activation_function2']
    optimizer=hyperparams['optimizer']
    module = __import__('kernel_initialization')
    KernelInit = getattr(module, hyperparams['init_weights'])

    model = NeuralNetwork(loss='mse', metric='mee')
    model.add_layer(dim_hid, input_dim=INPUT_DIM, activation=activation_function1, kernel_initialization=KernelInit())
    if(add_layer==True):
        dim_hid_2=int(hyperparams['hidden_nodes2'])
        activation_function2=hyperparams['activation_function2']
        model.add_layer(dim_hid_2,activation=activation_function2,kernel_initialization=KernelInit())
    model.add_layer(OUTPUT_DIM, activation='linear', kernel_initialization=KernelInit())

    if(optimizer=='SGD'):
        mom = hyperparams['alpha']
        model.compile(optimizer=SGD(lr=lr, mom=mom,l2=l2, nesterov=True))
    if (optimizer == 'RMSprop'):
        rho = hyperparams['rho']
        model.compile(optimizer=RMSprop(lr=lr, moving_average=rho, l2=l2))
    if (optimizer == 'Adam'):
        beta_1 = hyperparams['beta_1']
        beta_2 = hyperparams['beta_2']
        model.compile(optimizer=Adam(lr=lr, beta_1=beta_1,beta_2=beta_2, l2=l2))

    return model

start_time = time.time()

parser = Cup_parser(DIR_CUP + PATH_TR)
data, targets = parser.parse(INPUT_DIM, OUTPUT_DIM)

X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(data, targets, val_size=PERC_VL, test_size=PERC_TS, shuffle=True)
k_fold=1
#X_test,Y_test,folds_X,folds_Y=train_val_test_split_k_fold(data,targets,test_size=PERC_TS, shuffle=True,k_fold=k_fold)



model = random_search(
    create_model=create_model,
    tr=(X_train, Y_train),
    k_fold=k_fold,
    epochs=1000,
    batch_size=10,
    vl=(X_val,Y_val),
    ts=(X_test, Y_test),
    param_grid=PARAM_GRID,
    monitor_value='val_mee',
    n_threads=50,
    max_evals=200,
    path_results=path_result_randomsearch,
    tol=1e-3,
    verbose=True,
    shuffle=True
)

loss_tr, mee_tr = model.evaluate(X_train, Y_train)
loss_ts, mee_ts = model.evaluate(X_test, Y_test)
loss = [loss_tr, loss_ts]
mee = [mee_tr, mee_ts]

res = {
    'MSE': loss,
    'MEE': mee,
}

print(res)
print(model.evaluate(X_test, Y_test))
print(time.time() - start_time, "seconds")

write_results(res, model=None, save_result=path_result_model, test=False, validation=True)
