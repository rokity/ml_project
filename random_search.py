import numpy as np
import multiprocessing
import random
import pandas as pd


def print_hyperparams(hyperperams):
    """

    @param hyperperams: hyperparameters
    """
    for k, v in hyperperams.items():
        print("\t{:20}: {:10}".format(k, str(v)))


def write_csv(l_results, path, hyps_name, monitor_value):
    """

    @param l_results: list of random search results
    @param path: file csv
    @param hyps_name: hyperaparameters names
    @param monitor_value: criterion used to select the best model
    """
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


def run(model, tr, vl, ts, results, verbose, tol, epochs, batch_size, hyperparams, monitor_value, shuffle):
    """
    executes one evaluation with a fixed set of hyperparameters

    @return: validation error on the monitor_value
    """
    X_train, Y_train = tr
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, vl=vl, ts=ts, verbose=False, tol=tol, shuffle=shuffle)
    val = model.history[monitor_value][-1]
    results.append((val, hyperparams, model))
    if verbose:
        print("[+] Task completed")
    return val


def random_search(
        create_model,
        tr,
        vl,
        epochs,
        batch_size,
        param_grid,
        monitor_value='val_loss',
        ts=None, max_evals=10, path_results=None, n_threads=None, tol=None, verbose=False, shuffle=False
    ):
    """

    @param create_model: function used to create the neural network
    @param tr: pair (X_train, Y_train)
    @param vl: pair (X_val, Y_val)
    @param epochs: maximum number of epoch
    @param batch_size: size of batch
    @param param_grid: dictionary used to set the hyperparameters
    @param monitor_value: criterion used to select the best model
    @param ts: pair (X_test, Y_test) It is used only for plots
    @param max_evals: maximum number of attemps
    @param path_results: path used to write the random search result
    @param n_threads: number of threads
    @param tol: tolerance
    @param verbose: used for debug
    @param shuffle: True if you want shuffle training data (at each epoch),
                    False otherwise
    @return: best model under the monitor_value criterion
    """
    print('[+] Random search is started')

    if n_threads is None:
        n_threads = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_threads)
    results = multiprocessing.Manager().list()

    for i in range(max_evals):
        hyperaparams = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        model = create_model(hyperaparams)
        pool.apply_async(
            func=run,
            args=(model, tr, vl, ts, results, verbose, tol, epochs, batch_size, hyperaparams, monitor_value, shuffle)
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

    print('[+] Random search is finished')

    _, _, best_model = l_results[0]

    return best_model
