import numpy as np
import multiprocessing
import random
import pandas as pd


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
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, vl=vl, ts=ts, verbose=False, tol=tol)
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

    for i in range(max_evals):
        hyperaparams = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        model = create_model(hyperaparams)
        pool.apply_async(
            func=run,
            args=(model, tr, vl, ts, results, verbose, tol, epochs, batch_size, hyperaparams, monitor_value)
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





def main():
    PARAM_GRID = {
    'alpha': list(np.linspace(0.1, 1, 10).round(2)),
    'eta': list(np.linspace(0.1, 1, 10).round(2)),
    'hidden_nodes': list(np.linspace(2, 4, 3, dtype=np.uint8))
    }


if __name__ == "__main__":
    main()