import numpy as np
import multiprocessing
import random
import pandas as pd
from copy import deepcopy


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


def random_search(model, tr, vl, ts, max_evals=10, param_grid=None, path_results=None, n_threads=None, verbose=False):
    print('[+] Random search is started')
    if param_grid is None:
        param_grid = PARAM_GRID

    if n_threads is None:
        n_threads = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_threads)
    results = multiprocessing.Manager().list()

    for i in range(max_evals):
        hyperaparams = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
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

    print('[+] Random search is finished')

    return best_model

