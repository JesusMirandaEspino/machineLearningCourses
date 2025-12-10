# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 10:57:53 2025

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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline



dateset = pd.read_csv("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/housing.csv",  delim_whitespace=True, header=None)
data = dateset.values


X = data[:, 0:13].astype(float)
y = data[:, 13]


def create_model():
    model = Sequential()
    model.add( Dense(13, input_dim=13, activation="relu") )
    model.add( Dense(1) )
    model.summary()
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    return model

estimator = KerasRegressor(build_fn=create_model, epochs=100, batch_size=5)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, y, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)", (results.mean() * 100, results.std() * 100 ))




estimators = []
estimators.append(("standardize", StandardScaler()))
estimators.append(("NN", KerasRegressor(model=create_model, epochs=100, batch_size=5)))
pipeline = Pipeline(estimators)


kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, y, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))



def larger_model():
    model = Sequential()
    model.add( Dense(13, input_dim=13, activation="relu") )
    model.add( Dense(6, activation="relu") )
    model.add( Dense(1) )
    model.summary()
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    return model



estimators = []
estimators.append(("standardize", StandardScaler()))
estimators.append(("NN", KerasRegressor(model=larger_model, epochs=100, batch_size=5)))
pipeline = Pipeline(estimators)


kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, y, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))




def wider_model():
    model = Sequential()
    model.add( Dense(20, input_dim=13, activation="relu") )
    model.add( Dense(1) )
    model.summary()
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
    return model



estimators = []
estimators.append(("standardize", StandardScaler()))
estimators.append(("NN", KerasRegressor(model=wider_model, epochs=50, batch_size=5)))
pipeline = Pipeline(estimators)


kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))














