from parser import Cup_parser
from utility import *
from neural_network import NeuralNetwork
from kernel_initialization import *
from early_stopping import *
from optimizers import *

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
X_train, Y_train, X_test, Y_test = train_test_split(data, targets, test_size=PERC_TS, shuffle=True)

model = NeuralNetwork('mse', 'mee')

model.add_layer(30, input_dim=X_train.shape[1], activation='tanh', kernel_initialization=XavierNormalInitialization())
model.add_layer(30, activation='tanh', kernel_initialization=XavierNormalInitialization())
model.add_layer(Y_train.shape[1], activation='linear', kernel_initialization=XavierNormalInitialization())

model.compile(optimizer=RMSprop(lr=0.0001, moving_average=0.9, l2=0.0005))

model.fit(
    X_train, Y_train, 500, batch_size=64, ts=(X_test, Y_test),
    shuffle=True, tol=1e-2, verbose=True)
model.plot_loss(val=False, test=True)
model.plot_metric(val=False, test=True)
