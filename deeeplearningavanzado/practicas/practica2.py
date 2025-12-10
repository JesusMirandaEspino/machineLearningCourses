# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 09:54:32 2025

@author: jesus
"""

import pandas as pd
import tensorflow as ft
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier, KerasRegressor
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV


def create_model():
    model = Sequential()
    model.add( Dense(12, input_dim=8, activation="relu") )
    model.add( Dense(8, activation="relu") )
    model.add( Dense(1, activation="sigmoid") )
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model



def create_model_2(optimizer="rmsprop", init="glorot_uniform"):
    model = Sequential()
    model.add( Dense(12, input_dim=8, kernel_initializer=init,  activation="relu") )
    model.add( Dense(8, activation="relu") )
    model.add( Dense(1, activation="sigmoid") )
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

dateset = np.loadtxt("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/pima-indians-diabetes.csv", delimiter=",")
X = dateset[:, 0:8]
y = dateset[:, 8]


model = KerasClassifier(model= create_model_2, init='glorot_uniform', verbose=0)

optimizers = ["rmsprop", "adam"]
inits = ["glorot_uniform", "normal" "uniform"]
epochs = [50, 100, 150]
batchs = [6, 12, 16]
param_grid = dict( optimizer=optimizers, epochs=epochs, batch_size=batchs, init=inits )


grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X,y)

print(grid_result.best_score_, grid_result.best_params_)
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]


# Best 0.7044270833333334 {'batch_size': 6, 'epochs': 150, 'init': 'glorot_uniform', 'optimizer': 'rmsprop'}

for mean, std, param in zip(means, stds, params):
    print("------------------------------------")
    print(mean, std, param)
    print("------------------------------------")

kfolk = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = cross_val_score( model, X, y, cv=kfolk)
print(results.mean())
























































