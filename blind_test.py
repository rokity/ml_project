from neural_network import NeuralNetwork
from optimizers import *
from kernel_initialization import *
from functions_factory import *
from parser import Cup_parser
from utility import *


DIR_CUP = './data/cup/'
PATH_TR = 'ML-CUP19-TR.csv'
PATH_TS = 'ML-CUP19-TS.csv'
INPUT_DIM = 20
OUTPUT_DIM = 2

results=list()

parser = Cup_parser(DIR_CUP + PATH_TR)
data, targets = parser.parse(INPUT_DIM, OUTPUT_DIM)
X_train, Y_train,X_val,Y_val, X_test, Y_test = train_val_test_split(data, targets,val_size=0.25, test_size=0.25, shuffle=True)


model_1 = NeuralNetwork('mse', 'mee')
model_1.add_layer(30, input_dim=20, activation='sigm', kernel_initialization=RandomInitialization())
model_1.add_layer(40, activation="tanh", kernel_initialization=RandomInitialization())
model_1.add_layer(2, activation='linear', kernel_initialization=RandomInitialization())
optimizer = RMSprop(lr=0.001, moving_average=0.099, l2=0.0005)
model_1.compile(optimizer=optimizer)
model_1.fit(X_train, Y_train, early_stopping=None,epochs=500,
                  batch_size=32, vl=(X_val,Y_val), ts=(X_test,Y_test), verbose=False, tol=None, shuffle=True)

val = model_1.history['val_mee'][-1]
results.append((val, ['hyperparameters'], model_1))
