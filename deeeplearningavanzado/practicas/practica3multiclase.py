# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 08:25:25 2025

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
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scikeras.wrappers import KerasClassifier


dateset = pd.read_csv("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/iris.csv", header=None)
data = dateset.values
X = data[:, 0:4].astype(float)
y = data[:, 4]


encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

y_dummy = to_categorical(encoded_y)

def create_model():
    model = Sequential()
    model.add( Dense(8, input_dim=4, activation="relu") )
    model.add( Dense(3, activation="softmax") )
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model



estimator = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, y_dummy, cv=kfold)

print("Accuracy: %.2f%% {%.2f%%}", (results.mean() * 100, results.std() * 100 ))



























































