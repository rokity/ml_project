import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parser import Cup_parser
from utility import *
from neural_network import NeuralNetwork
from kernel_initialization import *

set_style_plot()

INPUT_DIM = 20
OUTPUT_DIM = 2


def build_model():
  model = NeuralNetwork(loss='mse', metric='mee')
  model.add_layer(60, input_dim=INPUT_DIM, activation='sigmoid',
                  kernel_initialization=XavierNormalInitialization())
  model.add_layer(30,  activation='sigmoid',
                  kernel_initialization=XavierNormalInitialization())
  model.add_output_layer(OUTPUT_DIM, activation='sigmoid',
                         kernel_initialization=XavierNormalInitialization())
  model.compile(lr=0.00588749, momentum=0.84150405, l2=0.00281810)
  return model


DIR_CUP = './data/cup/'
PATH_TR = 'ML-CUP19-TR.csv'
PATH_TS = 'ML-CUP19-TS.csv'
INPUT_DIM = 20
OUTPUT_DIM = 2

PERC_VL = 0.25
PERC_TS = 0.25

parser = Cup_parser(DIR_CUP + PATH_TR)
data, targets = parser.parse(INPUT_DIM, OUTPUT_DIM)
X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(
    data, targets, val_size=PERC_VL, test_size=PERC_TS, shuffle=True)
results = []
model = build_model()
model.fit(X_train, Y_train, epochs=1000, batch_size=100,
          vl=(X_val, Y_val), ts=(X_test, Y_test), verbose=True)
val = model.history['val_mee'][-1]
results.append((val, {}, model))
print("[+] Task completed")

l_results = list(results)
l_results.sort(key=lambda x: x[0])
_, best_hyps, best_model = l_results[0]
_, _, best_model = l_results[0]

best_model.plot_loss(val=False, test=True, show=True)
best_model.plot_metric(val=False, test=True, show=True)

err_tr, acc_tr = best_model.evaluate(X_train, Y_train)
err_ts, acc_ts = best_model.evaluate(X_test, Y_test)
errors = [err_tr, err_ts]
accuracy = [acc_tr, acc_ts]

res = {
    'Error': errors,
    'Accuracy': accuracy,
}

print(res)
path_err = 'out/cup/grid_search_err'
path_acc = 'out/cup/grid_search_acc'
path_result_bestmodel = 'out/cup/grid_search_results.csv'
save = (path_err, path_acc, path_result_bestmodel)
write_results(
    res, best_model,
    save_plot_loss=path_err, save_plot_metric=path_acc, save_result=path_result_bestmodel,
    validation=False,
    test=True,
    show=True
)
