# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 09:00:21 2025

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



dateset = pd.read_csv("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/sonar.csv",  header=None)
data = dateset.values


X = data[:, 0:60].astype(float)
y = data[:, 60]

encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)



def baseline_model():
    model = Sequential()
    model.add( Dense(60, input_dim=60, activation="relu", kernel_constraint=MaxNorm(max_value=3)) )
    model.add(Dropout(0.2))
    model.add( Dense(30, activation="relu", kernel_constraint=MaxNorm(max_value=3)) )
    model.add(Dropout(0.2))
    model.add( Dense(1, activation="sigmoid") )
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model



estimators = []
estimators.append(("standardize", StandardScaler()))
estimators.append(("MLP", KerasClassifier(model=baseline_model, epochs=300, batch_size=16)))
pipeline = Pipeline(estimators)

kfold_ = StratifiedKFold(n_splits=10, shuffle=True)

results = cross_val_score(pipeline, X, encoded_y, cv=kfold_)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))





