import pandas as pd


def write_results(res, best_model, save=None, all=False):
    results = pd.DataFrame(res, index=['Training set', 'Test set'])
    print(results)
    if all:
        best_model.show_all_err()
        best_model.show_all_acc()
    else:
        best_model.show_trts_err()
        best_model.show_trts_acc()
    if save is not None:
        path_err, path_acc, path_result_bestmodel = save
        results.to_csv(path_result_bestmodel, index=True)
        if all:
            best_model.save_all_err(path_err)
            best_model.save_all_acc(path_acc)
        else:
            best_model.save_trts_err(path_err)
            best_model.save_trts_acc(path_acc)
