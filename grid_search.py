import numpy as np
import multiprocessing
import random
import pandas as pd
from copy import deepcopy
from neural_network import NeuralNetwork
from parser import *
from utility import write_results


PARAM_GRID = {
    'alpha': list(np.linspace(0.1, 1, 10)),
    'eta': list(np.linspace(0.1, 1, 10)),
    'lambda': list(np.linspace(0.01, 0.1, 10))
}


def write_csv(l_results, path):
    err = []
    eta = []
    mom = []
    lam = []
    for (et, m, l, er, _) in l_results:
        err.append(er)
        eta.append(et)
        mom.append(m)
        lam.append(l)

    res = {
        'eta': eta,
        'momentum': mom,
        'lambda': lam,
        'error': err
    }

    results = pd.DataFrame(res)
    results.to_csv(path, index=False)


def run(nn, tr, vl, ts, results, verbose):
    err = nn.train(2000, tr, vl, ts, verbose=False)
    results.append((nn.eta, nn.alpha, nn.lam, err, nn))
    if verbose:
        print("[+] Task completed")
    return err


def grid_search(model, tr, vl, ts, max_evals=10, param_grid=None, path_results=None, n_threads=None, verbose=False):
    print('[+] Grid search is started')
    if param_grid is None:
        param_grid = PARAM_GRID

    if n_threads is None:
        n_threads = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_threads)
    results = multiprocessing.Manager().list()

    for i in range(max_evals):
        hyperaparams = {k: random.sample(v, 1)[0]
                        for k, v in param_grid.items()}
        eta = round(hyperaparams['eta'], 2)
        alpha = round(hyperaparams['alpha'], 2)
        lam = round(hyperaparams['lambda'], 2)
        nn = deepcopy(model)
        nn.compile(eta, alpha, lam, tr.size)
        pool.apply_async(func=run, args=(nn, tr, vl, ts, results, verbose))

    if verbose:
        print('[+] All threads are loaded')

    pool.close()
    pool.join()

    l_results = list(results)
    l_results.sort(key=lambda x: x[3])
    if verbose:
        for e, m, l, err, _ in l_results:
            print("[{}, {}, {}]:\t{}".format(e, m, l, err))

    if path_results is not None:
        write_csv(l_results, path_results)

    _, _, _, _, best_model = l_results[0]

    if verbose:
        print("Best:\n\t{:<15}{:>10}\n\t{:<15}{:>10}\n\t{:<15}{:>10}"
              .format(
                  'eta:',
                  best_model.eta,
                  'momentum:',
                  best_model.alpha,
                  'lamda:',
                  best_model.lam))

    print('[+] Grid search is finished')

    return best_model


def main():
    # initialize parameters
    path_tr = 'data/cup/ML-CUP19-TR.csv'
    path_ts = 'data/cup/ML-CUP19-TS.csv'
    path_result_randomsearch = 'out/cup/gridsearch.csv'
    path_err = 'out/cup/mse_cup'
    path_acc = 'out/cup/mee_cup'
    path_result_bestmodel = 'out/cup/results_gridsearch.csv'

    # activation functions
    f = 'sigmoid'
    out_f = 'linear'

    # loss function
    loss = 'mse'

    # accuracy function
    acc = 'mee'

    dim_in = 20
    dim_hid = 15
    dim_hid2 = 10
    dim_out = 2
    fan_in = dim_hid + dim_hid2

    parser = Cup_parser(path_tr)
    tr, vl, ts = parser.parse(dim_in, dim_out)
    model = NeuralNetwork(loss=loss, acc=acc)
    model.add_input_layer(dim_in, dim_hid, f, fan_in)
    model.add_hidden_layer(dim_hid2, f, fan_in)
    model.add_output_layer(dim_out, out_f, fan_in)

    best_model = grid_search(model, tr, vl, ts, max_evals=10,
                             path_results=path_result_randomsearch, verbose=True, n_threads=8)

    err_tr, acc_tr = best_model.predict_dataset(tr)
    err_ts, acc_ts = best_model.predict_dataset(ts)
    errors = [err_tr, err_ts]
    accuracy = [acc_tr, acc_ts]

    res = {
        'mse': errors,
        'mee': accuracy,
    }

    # save = (path_err, path_acc, path_result_bestmodel)
    # write_results(res, best_model, save=save, all=True)


if __name__ == "__main__":
    main()
