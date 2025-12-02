# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 09:09:07 2025

@author: jesus
"""

import pandas as pd
import tensorflow as ft
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", header=None)

# Tuplas
a = (1,2,3,4)


#listas
a = [1,2,3,4,5]

#Diccionarios
a = {"a": 5, "b": 6, "c": 7}

dateset = np.loadtxt("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/pima-indians-diabetes.csv", delimiter=",")
X = dateset[:, 0:8]
y = dateset[:, 8]




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

model = Sequential()
model.add( Dense(12, input_dim=8, activation="relu") )
model.add( Dense(8, activation="relu") )
model.add( Dense(1, activation="sigmoid") )


model.summary()


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X, y, validation_data=(X_test, y_test),  epochs=150, batch_size=16)

_, accuracy = model.evaluate(X, y)

predictions = model.predict(X)
print(predictions)

y_pred_classes = np.argmax(y)

print(y_pred_classes)



kfold = StratifiedKFold( n_splits=10, shuffle=True, random_state=42 )
cvscores = []
for train, test in kfold.split(X, y):
    model = Sequential()
    model.add( Dense(12, input_dim=8, activation="relu") )
    model.add( Dense(8, activation="relu") )
    model.add( Dense(1, activation="sigmoid") )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(X[train], y[train], epochs=150, batch_size=10, verbose=0)
    evaluate_model = model.evaluate(X[test], y[test], verbose=0)
    print("===========")
    print(model.metrics_names[1])
    print("===========")
    print(evaluate_model[1] * 100)
    cvscores.append(evaluate_model[1] * 100)

































































































