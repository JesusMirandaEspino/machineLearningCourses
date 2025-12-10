# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 09:56:17 2025

@author: jesus
"""


import pandas as pd
import tensorflow as ft
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
import math as mth

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD
from tensorflow.keras.constraints import MaxNorm
from scikeras.wrappers import KerasClassifier, KerasRegressor

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline
from tensorflow.keras.optimizers.schedules import PolynomialDecay


#ionosphere
dateset = pd.read_csv("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/Datasets/ionosphere.csv",  header=None)
data = dateset.values

X = data[:, 0:34].astype(float)
y = data[:, 34]


encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)



def step_decay(epoch):
    initial_rate = 0.1
    drop = 0.5
    epochs_drop = 10
    lrate = initial_rate * mth.pow(drop,  mth.floor( 1 + epoch / epochs_drop))
    return lrate




model = Sequential()
model.add( Dense(34, input_dim=34, activation="relu") )
model.add( Dense(1, activation="sigmoid") )

epochs = 50
lr = 0.1
decay_rate = lr / epochs
momentum = 0.8


initial_lr = 0.1

# Decae linealmente hasta 0
lr_schedule = PolynomialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=epochs,        # número de pasos (puedes usar steps_per_epoch * epochs)
    end_learning_rate=0.0,     # valor final
    power=1.0                  # 1.0 = lineal
)


sgd = SGD(learning_rate=0.0, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])

lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

history = model.fit( X, encoded_y, validation_split=0.33, epochs=50, batch_size=28, callbacks=callbacks_list, verbose=2 )




