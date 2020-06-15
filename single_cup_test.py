from parser import Cup_parser
from utility import *
from neural_network import NeuralNetwork
from kernel_initialization import *
from early_stopping import *

DIR_CUP = './data/cup/'
PATH_TR = 'ML-CUP19-TR.csv'
PATH_TS = 'ML-CUP19-TS.csv'
INPUT_DIM = 20
OUTPUT_DIM = 2

PERC_VL = 0.25
PERC_TS = 0.25

set_style_plot()

parser = Cup_parser(DIR_CUP + PATH_TR)
data, targets = parser.parse(INPUT_DIM, OUTPUT_DIM)
X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(data, targets, val_size=PERC_VL, test_size=PERC_TS, shuffle=True)

model = NeuralNetwork('mse', 'mee')

model.add_layer(40, input_dim=X_train.shape[1], activation='sigmoid', kernel_initialization=RandomUniformInitialization())
model.add_output_layer(Y_train.shape[1], activation='linear', kernel_initialization=RandomUniformInitialization())

model.compile(lr=0.05, momentum=0.8, l2=0.001, tau=300, perc_eps_t=1)

model.fit(
    X_train, Y_train, 500, batch_size=16, vl=(X_val, Y_val), ts=(X_test, Y_test),
    shuffle=True, tol=1e-2, verbose=True)
model.plot_loss(val=True, test=True)
model.plot_metric(val=True, test=True)
