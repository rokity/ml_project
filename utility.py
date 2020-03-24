import pandas as pd


def write_results(res, best_model, save=None):
    results = pd.DataFrame(res, index=['Training set', 'Test set'])
    print(results)
    best_model.show_all_err()
    best_model.show_all_acc()
    if save is not None:
        path_err, path_acc, path_result_bestmodel = save
        results.to_csv(path_result_bestmodel, index=True)
        best_model.save_all_err(path_err)
        best_model.save_all_acc(path_acc)