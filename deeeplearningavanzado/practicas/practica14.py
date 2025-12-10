# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 09:20:20 2025

@author: jesus
"""

import pandas as pd
import tensorflow as ft
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
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


sgd = SGD(learning_rate=lr_schedule, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])



history = model.fit( X, encoded_y, validation_split=0.33, epochs=epochs, batch_size=28, verbose=2 )






































































































