from neural_network import NeuralNetwork
from parser import Monks_parser
from  utility import set_style_plot
from kernel_initialization import *
from utility import write_results
from utility import change_output_value
import pandas as pd
import numpy as np
import multiprocessing
import random
import pandas as pd

set_style_plot()

PARAM_GRID = {
        'alpha': list(np.linspace(0.1, 0.9, 9).round(2)),
        'eta': list(np.linspace(0.1, 0.9, 9).round(2)),
        'hidden_nodes': list(np.linspace(2, 4, 3, dtype=np.uint8))
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
              vl=vl, ts=ts, verbose=False, tol=tol)
    val = model.history[monitor_value][-1]
    results.append((val, hyperparams, model))
    if verbose:
        print("[+] Task completed")
    return val


def grid_search(
    create_model,
    tr,
    vl,
    epochs,
    batch_size,
    param_grid,
    monitor_value='val_loss',
    ts=None, max_evals=10, path_results=None, n_threads=None, tol=None, verbose=False
):
    print('[+] Grid search is started')

    if n_threads is None:
        n_threads = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_threads)
    results = multiprocessing.Manager().list()

    for 
    # for i in range(max_evals):
    #     hyperaparams = {k: random.sample(v, 1)[0]
    #                     for k, v in param_grid.items()}
    #     model = create_model(hyperaparams)
    #     pool.apply_async(
    #         func=run,
    #         args=(model, tr, vl, ts, results, verbose, tol,
    #               epochs, batch_size, hyperaparams, monitor_value)
    #     )

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


path_tr = 'data/monks/monks-1.train'
path_ts = 'data/monks/monks-1.test'
path_result_randomsearch = 'out/monks/monk1/randomsearch.csv'
path_err = 'out/monks/monk1/err_monk1_gridsearch'
path_acc = 'out/monks/monk1/acc_monk1_gridsearch'
path_result_bestmodel = 'out/monks/monk1/results_gridsearch.csv'

dim_in = 6
one_hot = 17
dim_out = 1

parser = Monks_parser(path_tr, path_ts)

X_train, Y_train, X_test, Y_test = parser.parse(dim_in, dim_out, one_hot)

Y_train = change_output_value(Y_train, 0, -1)
Y_test = change_output_value(Y_test, 0, -1)

dim_in = one_hot




def create_model(hyperparams):
    lr = hyperparams['eta']
    mom = hyperparams['alpha']
    dim_hid = int(hyperparams['hidden_nodes'])

    model = NeuralNetwork(loss='mse', metric='accuracy1-1')
    model.add_layer(dim_hid, input_dim=dim_in, activation='sigmoid', kernel_initialization=XavierNormalInitialization())
    model.add_output_layer(dim_out, activation='tanh', kernel_initialization=XavierUniformInitialization())

    model.compile(lr, mom)

    return model


def main():
    
    model = grid_search(
        create_model,
        (X_train, Y_train),
        (X_train, Y_train),
        20,
        X_train.shape[0],
        param_grid=PARAM_GRID,
        monitor_value='mse',
        ts=(X_test, Y_test),
        max_evals=5,
        n_threads=1,
        #path_results=path_result_randomsearch,
        tol=0.5,
        verbose=True
    )

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
    save = (path_err, path_acc, path_result_bestmodel)
    write_results(res, model, save=save,all=True)


if __name__ == "__main__":
    main()
