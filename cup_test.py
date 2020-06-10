from parser import Cup_parser
from utility import *
from neural_network import NeuralNetwork
from kernel_initialization import *

DIR_CUP = './data/cup/'
PATH_TR = 'ML-CUP19-TR.csv'
PATH_TS = 'ML-CUP19-TS.csv'
INPUT_DIM = 20
OUTPUT_DIM = 2

PERC_VL = 0.25
PERC_TS = 0.20

parser = Cup_parser(DIR_CUP + PATH_TR)
data, targets = parser.parse(INPUT_DIM, OUTPUT_DIM)
X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(data, targets, val_size=PERC_VL, test_size=PERC_TS, shuffle=True)

model = NeuralNetwork('mse', 'mee')

model.add_layer(40, input_dim=X_train.shape[1], activation='sigmoid', kernel_initialization=RandomUniformInitialization())
model.add_output_layer(Y_train.shape[1], activation='linear', kernel_initialization=RandomUniformInitialization())

model.compile(lr=1e-3)

model.fit(X_train, Y_train, 100, X_train.shape[0], vl=(X_val, Y_val), ts=(X_test, Y_test), tol=1e-2, verbose=True)
model.plot_loss(val=True, test=False)
model.plot_metric(val=True, test=False)
