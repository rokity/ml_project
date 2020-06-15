from parser import Cup_parser
from utility import *
from neural_network import NeuralNetwork
from kernel_initialization import *
from random_search import random_search


DIR_CUP = './data/cup/'
PATH_TR = 'ML-CUP19-TR.csv'
PATH_TS = 'ML-CUP19-TS.csv'
path_result_randomsearch = "./out/cup/randomsearch2.csv"
path_result_model = "./out/cup/results.csv"
INPUT_DIM = 20
OUTPUT_DIM = 2

PERC_VL = 0.25
PERC_TS = 0.15


PARAM_GRID = {
    'eta': list(np.linspace(0.01, 0.09, 9).round(3)) + list(np.linspace(0.001, 0.009, 9).round(4)),
    'alpha': list(np.linspace(0.1, 0.9, 9).round(2)),
    'lambda': list(np.linspace(0.001, 0.009, 9).round(5)),
    'tau': list([200, 300, 400]),
    'perc_epst': list([1, 2, 3]),
    'add_layer': list([True, False, False]),
    'hidden_nodes': list([10, 15, 20, 30, 40]),
    'hidden_nodes2': list([10, 15, 20, 30, 40])
}

for k, v in PARAM_GRID.items():
    print(k, ",  ", v)


def create_model(hyperparams):
    lr = hyperparams['eta']
    mom = hyperparams['alpha']
    l2 = hyperparams['lambda']
    tau = int(hyperparams['tau'])
    perc_epst = int(hyperparams['perc_epst'])
    dim_hid = int(hyperparams['hidden_nodes'])
    dim_hid2 = int(hyperparams['hidden_nodes2'])
    add_layer = hyperparams['add_layer']

    model = NeuralNetwork(loss='mse', metric='mee')
    model.add_layer(dim_hid, input_dim=INPUT_DIM, activation='sigmoid', kernel_initialization=RandomNormalInitialization())
    if add_layer:
        model.add_layer(dim_hid2, activation='sigmoid', kernel_initialization=RandomNormalInitialization())
    model.add_output_layer(OUTPUT_DIM, activation='linear', kernel_initialization=RandomNormalInitialization())

    model.compile(lr=lr, momentum=mom, l2=l2, tau=tau, perc_eps_t=perc_epst)

    return model


parser = Cup_parser(DIR_CUP + PATH_TR)
data, targets = parser.parse(INPUT_DIM, OUTPUT_DIM)
X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(data, targets, val_size=PERC_VL, test_size=PERC_TS, shuffle=True)

model = random_search(
    create_model,
    (X_train, Y_train),
    (X_val, Y_val),
    1000,
    batch_size=16,
    #ts=(X_test, Y_test),
    param_grid=PARAM_GRID,
    monitor_value='val_mee',
    n_threads=8,
    max_evals=30,
    path_results=path_result_randomsearch,
    tol=1e-3,
    verbose=True,
    shuffle=True
)

model.plot_loss(val=False, test=True, show=True)
model.plot_metric(val=False, test=True, show=True)

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
write_results(res, model=None, save_result=path_result_model, test=False, validation=True)