from functions_factory import FunctionsFactory
from neural_network import NeuralNetwork
from parser import *
import numpy as np
import multiprocessing
import queue
import random


def run(nn, tr, vl, ts, results):
    err = nn.train(tr, tr, ts, 1e-2, 2000)
    results.append((err, nn.get_eta(), nn.get_momentum(), nn.get_lambda()))
    return err


path_tr = 'monks/monks-1.train'
path_ts = 'monks/monks-1.test'
dim_in = 6
one_hot = 17
dim_hid = 4
dim_out = 1
MAX_EVALS = 16

f = FunctionsFactory.build('sigmoid')
loss = FunctionsFactory.build('lms')

acc = FunctionsFactory.build('accuracy')

if one_hot is None:
    topology = [dim_in, dim_hid, dim_out]
else:
    topology = [one_hot, dim_hid, dim_out]

parser = Monks_parser(path_tr, path_ts)

tr, vl, ts = parser.parse(dim_in, dim_out, one_hot, None)


param_grid = {
    'alpha': list(np.linspace(0.1, 1, 10)),
    'eta': list(np.linspace(0.1, 1, 10)),
    'lambda': list(np.linspace(0.01, 0.1, 10))
}

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()-1)
results = multiprocessing.Manager().list()

for i in range(MAX_EVALS):
    hyperaparams = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
    eta = round(hyperaparams['eta'], 2)
    alpha = round(hyperaparams['alpha'], 2)
    lam = round(hyperaparams['lambda'], 2)
    nn = NeuralNetwork(topology, f, loss, acc, dim_hid, tr.size, eta, alpha, lam)
    pool.apply_async(func=run, args=(nn, tr, vl, ts, results))

print('[+] All threads are loaded')

pool.close()
pool.join()

l_results = list(results)
l_results.sort(key=lambda x: x[0])
for res, e, m, l in l_results:
    print("[{}, {}, {}]:\t{}".format(e, m, l, res))
