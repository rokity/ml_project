import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parser import Cup_parser
from utility import *
from neural_network import NeuralNetwork
from kernel_initialization import *

import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

set_style_plot()



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



def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='sigmoid', input_shape=[20]),
    layers.Dense(64, activation='sigmoid'),
    layers.Dense(2)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse',tf.keras.metrics.RootMeanSquaredError()])
  return model


model = build_model()
model.summary()


EPOCHS = 400

history = model.fit(
  X_train, Y_train,
  epochs=EPOCHS,
  validation_data=(X_val,Y_val),
   verbose=2,
  )

print("--------------------------------------------")
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())
print("--------------------------------------------")


loss, mae, mse,mee = model.evaluate(X_test, Y_test, verbose=2)


print("Testing set Loss: {:5.2f} \n".format(loss))
print("Testing set Mae: {:5.2f} \n".format(mae))
print("Testing set Mse: {:5.2f} \n".format(loss))
print("Testing set Mee: {:5.2f} \n".format(mee))