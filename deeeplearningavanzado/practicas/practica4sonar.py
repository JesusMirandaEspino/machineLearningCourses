# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 09:07:51 2025

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


dateset = pd.read_csv("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/sonar.csv", header=None)
data = dateset.values

X = data[:, 0:60].astype(float)
y = data[:, 60]


encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

def create_model():
    model = Sequential()
    model.add( Dense(60, input_dim=60, activation="relu") )
    model.add( Dense(1, activation="sigmoid") )
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model



estimator = KerasClassifier(build_fn=create_model, epochs=100, batch_size=5)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, encoded_y, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)", (results.mean() * 100, results.std() * 100 ))
