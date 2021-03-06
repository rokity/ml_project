from neural_network import NeuralNetwork
from parser import Monks_parser
from utility import set_style_plot
from utility import *
from kernel_initialization import *
from optimizers import *

set_style_plot()

DIR_DATA = "./data/monks/"
TR_FILE = 'monks-2.train'
TS_FILE = 'monks-2.test'

path_tr = DIR_DATA + TR_FILE
path_ts = DIR_DATA + TS_FILE

dim_in = 6
one_hot = 17
dim_out = 1

parser = Monks_parser(path_tr, path_ts)

X_train, Y_train, X_test, Y_test = parser.parse(dim_in, dim_out, one_hot)

Y_train = change_output_value(Y_train, 0, -1)
Y_test = change_output_value(Y_test, 0, -1)

dim_in = one_hot
dim_hid = 2
model = NeuralNetwork('mse', 'accuracy1-1')

model.add_layer(dim_hid, input_dim=dim_in, activation='sigmoid',
                kernel_initialization=RandomUniformInitialization())
model.add_layer(dim_out, activation='tanh', kernel_initialization=RandomUniformInitialization(-1, 1))

model.compile(optimizer=SGD(lr=0.5, mom=0.8))
history = model.fit(X_train, Y_train, 120, X_train.shape[0], ts=(X_test, Y_test),
                    verbose=True
                    )

model.plot_loss(test=True)
