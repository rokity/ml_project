from neural_network import NeuralNetwork
from optimizers import *
from kernel_initialization import *
from functions_factory import *
from parser import Cup_parser
from utility import *
import multiprocessing
import random
import time
import os
import itertools


DIR_CUP = './data/cup/'
PATH_TR = 'ML-CUP19-TR.csv'
PATH_TS = 'ML-CUP19-TS.csv'
PATH_PARAMS='./params.csv'
path_result_randomsearch='./out/cup/grid_search.csv'
path_result_model='./out/cup/grid_search_model.csv'
path_plot_loss=os.getcwd()+'/out/cup/grid_search_plot_loss.png'
path_plot_metric=os.getcwd()+'/out/cup/grid_search_plot_metric.png'


INPUT_DIM = 20
OUTPUT_DIM = 2
PERC_TEST = 0.25
PERC_VAL= 0.25
LOSS = 'mse'
METRIC = 'mee'
K = 2




PARAM_SEARCH = {
    'eta': list([0.001, 0.0001,0.01]),
    'alpha': list([0.7,0.8,0.9]),
    'lambda': list([0.0002,0.0004,0.0005,0.0009]),
    'add_layer': list([True, False]),
    'hidden_nodes': list([ 20, 30, 40]),
    'hidden_nodes2': list([20, 30, 40]),
    'optimizer':list(['SGD','Adam','RMSprop']),
    'activation_function1':list(['tanh','sigmoid']),
    'activation_function2':list(['tanh','sigmoid']),
    'init_weights':list(['XavierNormalInitialization','HeInitialization','RandomNormalInitialization']),
    'batch_size':list([ 8, 16, 32, 64]),
    'beta_1': list([0.9,0.99,0.999]),
    'beta_2': list([0.9,0.99,0.999]),
}


def print_hyperparams(hyperperams):
    for k, v in hyperperams.items():
        print("\t{:20}: {:10}".format(k, str(v)))

def write_csv(l_results, path, hyps_name, monitor_value):
    res = dict()
    for hyp in hyps_name:
        res[hyp] = []
    res[monitor_value] = []

    for (val, hyps, _) in l_results:
        for k, v in hyps.items():
            res[k].append(str(v))
        res[monitor_value].append(val)

    results = pd.DataFrame(res)
    results.to_csv(path, index=False)


def run(model, tr, vl, ts, results, verbose, tol, epochs, batch_size, hyperparams, monitor_value, shuffle,k_folds=None,
        current_task =None,num_task=None):
    if verbose:
        print("[+] Start one task")


    if(k_folds==None):
        X_train, Y_train = tr
        model.fit(X_train, Y_train, early_stopping=None,epochs=epochs,
                  batch_size=batch_size, vl=vl, ts=ts, verbose=False, tol=tol, shuffle=shuffle)
        val = model.history[monitor_value][-1]
        results.append((val, hyperparams, model))
        if verbose:
            print("[+] Task completed")
        print("val : {}".format(val))
        return val
    else:
        folds_X,folds_Y=tr
        _model=model
        folds_result=list()
        for j in range(0, k_folds):
            vl_fold = (folds_X[j], folds_Y[j])
            x_tr_list = folds_X[:j + 1] + folds_X[j + 1:]
            y_tr_list = folds_Y[:j + 1] + folds_Y[j + 1:]
            model=_model
            model.fit(np.concatenate(x_tr_list), np.concatenate(y_tr_list), early_stopping=None, epochs=epochs,
                      batch_size=batch_size, vl=vl_fold, ts=ts, verbose=False, tol=tol, shuffle=shuffle)
            val = model.history[monitor_value][-1]
            folds_result.append((val, hyperparams, model))

        folds_result.sort(key=lambda x: x[0])
        best_val, best_hyps, best_model = folds_result[0]
        results.append((best_val, best_hyps, best_model))
        if verbose:
            print("[+] Task completed {}/{}".format(current_task,num_task))

        print("val : {}".format(val))
        return best_val





def random_search(
        create_model,
        tr,
        k_fold,
        epochs,
        batch_size,
        param_grid,
        monitor_value='val_loss',
        vl=None,
        ts=None,
        max_evals=10,
        path_results=None,
        n_threads=None,
        tol=None,
        verbose=False,
        shuffle=False
    ):

    print('[+] Random search is started')
    X_train, Y_train = tr

    if n_threads is None:
        n_threads = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_threads)
    results = multiprocessing.Manager().list()
    thread_list=list()


    for i in range(max_evals):
        hyperaparams = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        model = create_model(hyperaparams)
        thread_list.append(pool.apply_async(
                    func=run,
                    args=(model, tr, vl, ts, results, verbose, tol, epochs, hyperaparams['batch_size'], hyperaparams, monitor_value, shuffle,k_fold)
                ))

    if verbose:
        print('[+] All threads are loaded')

    pool.close()
    pool.join()
    [result.get() for result in thread_list]
    l_results = list(results)
    l_results.sort(key=lambda x: x[0])

    if verbose:
        for val, hyperaparams, nn in l_results:
            print("{}: {}".format(monitor_value, val))

    if path_results is not None:
        write_csv(l_results, path_results, param_grid.keys(), monitor_value)

    _, best_hyps, best_model = l_results[0]

    if verbose:
        print("Best result with: ")
        print_hyperparams(best_hyps)

    print('[+] Random search is finished')

    _, _, best_model = l_results[0]

    return best_model


def grid_search(create_model,
        tr,
        k_fold,
        epochs,
        batch_size,
        param_grid,
        monitor_value='val_loss',
        vl=None,
        ts=None,
        path_results=None,
        n_threads=None,
        tol=None,
        verbose=False,
        shuffle=False
        ):
    print('[+] Grid search is started')
    X_train, Y_train = tr

    if n_threads is None:
        n_threads = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_threads)
    results = multiprocessing.Manager().list()
    thread_list = list()
    num_task=len(param_grid)
    current_task = 0
    for row in param_grid:
            hyperaparams = {list(item.keys())[0]:list(item.values())[0]  for item in row }
            model = create_model(hyperaparams)
            current_task = current_task + 1
            thread_list.append(pool.apply_async(
                        func=run,
                        args=(model, tr, vl, ts, results, verbose, tol, epochs, hyperaparams['batch_size'], hyperaparams, monitor_value, shuffle,k_fold,current_task,num_task)
                    ))

    if verbose:
        print('[+] All threads are loaded')

    pool.close()
    pool.join()
    [result.get() for result in thread_list]
    l_results = list(results)
    l_results.sort(key=lambda x: x[0])

    if verbose:
        for val, hyperaparams, nn in l_results:
            print("{}: {}".format(monitor_value, val))

    if path_results is not None:
        write_csv(l_results, path_results, PARAM_SEARCH.keys(), monitor_value)

    _, best_hyps, best_model = l_results[0]

    if verbose:
        print("Best result with: ")
        print_hyperparams(best_hyps)

    print('[+] Random search is finished')

    _, _, best_model = l_results[0]

    return best_model



def create_model(hyperparams):
    _module_kernelinit = __import__('kernel_initialization')

    lr = hyperparams['eta']
    mom = hyperparams['alpha']
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
#X_train, Y_train, X_test, Y_test = train_test_split(data, targets, test_size=PERC_TEST, shuffle=True)
#X_train, Y_train,X_val,Y_val, X_test, Y_test = train_val_test_split(data, targets,val_size=PERC_VAL, test_size=PERC_TEST, shuffle=True)
X_test,Y_test,folds_X,folds_Y=train_val_test_split_k_fold(data,targets,test_size=PERC_TEST, shuffle=True,k_fold=K)

PARAM_GRID = [[{key: value} for (key, value) in zip(PARAM_SEARCH, values)]
                      for values in itertools.product(*PARAM_SEARCH.values())]
PARAM_GRID=PARAM_GRID[:5]
model=grid_search(
    create_model=create_model, tr=(folds_X, folds_Y), k_fold=K, epochs=500,
    batch_size=None, param_grid=PARAM_GRID, monitor_value='val_mee',
    vl=None,ts=(X_test,Y_test),
    path_results=path_result_randomsearch, verbose=True, shuffle=True,n_threads=4,
)

X_train, Y_train,X_val,Y_val, X_test, Y_test = train_val_test_split(data, targets,val_size=PERC_VAL, test_size=PERC_TEST, shuffle=True)
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
model.plot_loss(val=True, test=True,show=False,path=path_plot_loss)
model.plot_metric(val=True, test=True,show=False,path=path_plot_metric)




