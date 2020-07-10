from utility import *
import multiprocessing
import random


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


def run(create_model, tr, vl, ts, results, verbose, tol, epochs,
        batch_size, hyperparams, monitor_value, shuffle, k_folds=None, enum_task=None):
    """
    executes one evaluation with a fixed set of hyperparameters
    if k_folds is None it uses K-fold cross validation as validation tecnhique
    otherwise it uses a simple hold out using the validation set vl

    @return: validation error on the monitor_value
    """
    if verbose:
        print("[+] Start one task")

    if k_folds is None:
        model = create_model(hyperparams)
        X_train, Y_train = tr
        model.fit(X_train, Y_train, early_stopping=None,epochs=epochs,
                  batch_size=batch_size, vl=vl, ts=ts, verbose=False, tol=tol, shuffle=shuffle)
        val = model.history[monitor_value][-1]
        results.append((val, hyperparams, model))
        print("Hold out validation {} : {}".format(monitor_value, val))
        return val
    else:
        folds_X, folds_Y = tr
        folds_result = np.zeros(4)
        model = create_model(hyperparams)
        for j in range(0, k_folds):
            model = create_model(hyperparams)
            vl = (folds_X[j], folds_Y[j])
            X_train = np.concatenate(folds_X[0:j] + folds_X[j+1:])
            Y_train = np.concatenate(folds_Y[0:j] + folds_Y[j+1:])
            model.fit(X_train, Y_train, early_stopping=None, epochs=epochs,
                      batch_size=batch_size, vl=vl, ts=ts, verbose=False, tol=tol, shuffle=shuffle)
            val = model.history[monitor_value][-1]
            folds_result[j] = val

        mean_kfold_val = folds_result.mean()
        results.append((mean_kfold_val, hyperparams, model))

        print("Mean k-fold validation {} : {}".format(monitor_value, mean_kfold_val))

        if verbose:
            if enum_task is None:
                print("[+] Task completed ")
            else:
                curr_task, tot_task = enum_task
                print("[+] Task completed {}/{}".format(curr_task, tot_task))

        return mean_kfold_val


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
    """

    @param create_model: function used to create the neural network
    @param tr: pair (X_train, Y_train)
    @param k_fold: number of folds used for K-fold cross validation technique
    @param epochs: maximum number of epoch
    @param batch_size: size of batch
    @param param_grid: dictionary used to set the hyperparameters
    @param monitor_value: criterion used to select the best model
    @param vl: pair (X_val, Y_val)
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

    current_task = 1
    for i in range(max_evals):
        hyperaparams = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        pool.apply_async(
            func=run,
            args=(create_model, tr, vl, ts, results, verbose, tol, epochs, hyperaparams['batch_size'],
                  hyperaparams, monitor_value, shuffle, k_fold, (current_task, max_evals))
        )
        current_task += 1

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


def grid_search(
        create_model,
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
    """

    @param create_model: function used to create the neural network
    @param tr: pair (X_train, Y_train)
    @param k_fold: number of folds used for K-fold cross validation technique
    @param epochs: maximum number of epoch
    @param batch_size: size of batch
    @param param_grid: dictionary used to set the hyperparameters
    @param monitor_value: criterion used to select the best model
    @param vl: pair (X_val, Y_val)
    @param ts: pair (X_test, Y_test) It is used only for plots
    @param path_results: path used to write the random search result
    @param n_threads: number of threads
    @param tol: tolerance
    @param verbose: used for debug
    @param shuffle: True if you want shuffle training data (at each epoch),
                    False otherwise
    @return: best model under the monitor_value criterion
    """
    print('[+] Grid search is started')
    X_train, Y_train = tr

    if n_threads is None:
        n_threads = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_threads)
    results = multiprocessing.Manager().list()

    num_task = len(param_grid)
    current_task = 1

    for row in param_grid:
            hyperaparams = {list(item.keys())[0]: list(item.values())[0] for item in row}
            pool.apply_async(
                func=run,
                args=(create_model, tr, vl, ts, results, verbose, tol, epochs, hyperaparams['batch_size'],
                      hyperaparams, monitor_value, shuffle, k_fold, (current_task, num_task))
            )
            current_task += 1

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