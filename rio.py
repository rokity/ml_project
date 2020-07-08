from neural_network import NeuralNetwork
from parser import Cup_parser
from utility import set_style_plot
from kernel_initialization import *
from utility import write_results
from utility import change_output_value
import multiprocessing
import random
import pandas as pd
from utility import *
import importlib
import time 


set_style_plot()

INPUT_DIM=20
OUTPUT_DIM=2

PARAM_GRID = {
    'alpha': list(np.linspace(0.1, 0.9, 9).round(2)),
    'eta': list(np.linspace(0.001, 0.009, 9).round(4)),
    'lambda': list(np.linspace(0.0001, 0.0009, 9).round(5)),
    'functions':list(['sigmoid','tanh']),
    'hidden_nodes': list([20, 30, 40,50,60,70,80,90,100]),
    'batch_size':list([1, 2, 8, 16,32,10,10,20,30,40,50]),
    'kernel_init': list(['RandomInitialization','RandomNormalInitialization',
    'RandomUniformInitialization','XavierUniformInitialization',
    'XavierNormalInitialization','ZerosInitialization'])
}


def print_hyperparams(hyperperams):
    for k, v in hyperperams.items():
        print("\t{:20}: {:10}".format(k, v))


def write_csv(l_results, path, hyps_name, monitor_value):
    res = dict()
    for hyp in hyps_name:
        res[hyp] = []
    res[monitor_value] = []

    for (val, hyps, _) in l_results:
        for k, v in hyps.items():
            res[k].append(v)
        res[monitor_value].append(val)

    results = pd.DataFrame(res)
    results.to_csv(path, index=False)


def run(model, tr, vl, ts, results, verbose, tol, epochs, batch_size, hyperparams, monitor_value):
    X_train, Y_train = tr
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
              vl=vl, ts=ts, verbose=True, tol=tol)
    val = model.history[monitor_value][-1]
    results.append((val, hyperparams, model))
    if verbose:
        print("[+] Task completed")
    return val

def create_model(hyperparams):
    lr = hyperparams['eta']
    mom = hyperparams['alpha']
    l2 = hyperparams['lambda']
    dim_hid = int(hyperparams['hidden_nodes'])
    _fun = hyperparams['functions']
    kernel_init = hyperparams['kernel_init']
    module = __import__('kernel_initialization')
    
    KernelInit = getattr(module, kernel_init)
    
    model = NeuralNetwork(loss='mse', metric='mee')
    model.add_layer(dim_hid, input_dim=INPUT_DIM, activation=_fun, kernel_initialization=KernelInit())
    model.add_output_layer(2, activation=_fun, kernel_initialization=KernelInit())
    
    model.compile(lr=lr, momentum=mom, l2=l2)
    return model


def grid_search(
    create_model,
    tr,
    vl,
    epochs,
    batch_size,
    param_grid,
    monitor_value='val_mee',
    ts=None, max_evals=10, path_results=None, n_threads=None, tol=None, verbose=False
):
    print('[+] Grid search is started')

    if n_threads is None:
        n_threads = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_threads)
    results = multiprocessing.Manager().list()


    
    for i in range(1,3):                         
        hyperaparams = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        model = create_model(hyperaparams)
        pool.apply_async(func=run,
                                    args=(model, tr, vl, ts, results, True, tol,
                                                                    500, hyperaparams['batch_size'], hyperaparams, monitor_value)
                                                        )

    if verbose:
        print('[+] All threads are loaded')

    pool.close()
    pool.join()

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

    print('[+] Grid search is finished')

    _, _, best_model = l_results[0]

    return best_model



DIR_CUP = './data/cup/'
PATH_TR = 'ML-CUP19-TR.csv'
PATH_TS = 'ML-CUP19-TS.csv'
INPUT_DIM = 20
OUTPUT_DIM = 2

PERC_VL = 0.25
PERC_TS = 0.25

parser = Cup_parser(DIR_CUP + PATH_TR)
data, targets = parser.parse(INPUT_DIM, OUTPUT_DIM)
X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(data, targets, val_size=PERC_VL, test_size=PERC_TS, shuffle=True)







def main():
    
    model = grid_search(
        create_model,
        (X_train, Y_train),
        (X_val, Y_val),
        300,
        10,
        param_grid=PARAM_GRID,
        monitor_value='val_mee',
        ts=(X_test, Y_test),
        max_evals=100,
        n_threads=5,
        path_results='out/cup/grid_search_results.csv',
        verbose=True
    )

    model.plot_loss(val=False, test=True, show=True)
    model.plot_metric(val=False, test=True, show=True)

    err_tr, acc_tr = model.evaluate(X_train, Y_train)
    err_ts, acc_ts = model.evaluate(X_test, Y_test)
    errors = [err_tr, err_ts]
    accuracy = [acc_tr, acc_ts]

    res = {
        'Error': errors,
        'Accuracy': accuracy,
    }

    print(res)
    path_err='out/cup/grid_search_err'
    path_acc='out/cup/grid_search_acc'
    path_result_bestmodel='out/cup/grid_search_best_results.csv'
    save = (path_err, path_acc, path_result_bestmodel)
    write_results(
    res, model,
    save_plot_loss=path_err, save_plot_metric=path_acc, save_result=path_result_bestmodel,
    validation=False,
    test=True,
    show=True
)


if __name__ == "__main__":
    main()

