# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 09:08:43 2025

@author: jesus
"""

import pandas as pd
import tensorflow as ft
import numpy as np
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier, KerasRegressor
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline


dateset = np.loadtxt("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/pima-indians-diabetes.csv",  delimiter=",")

X = dateset[:, 0:8].astype(float)
y = dateset[:, 8]


model = Sequential()
model.add( Dense(12, input_dim=8, activation="relu") )
model.add( Dense(8, activation="relu") )
model.add( Dense(1, activation="sigmoid") )
model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


model.fit( X,y, epochs=150, batch_size=10 )
scores = model.evaluate(X,y)

print( model.metrics_names[1], scores[1] * 100 )

model.save("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/model.h5")


model_ = load_model("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/model.h5")
model_.summary()


dateset = np.loadtxt("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/pima-indians-diabetes.csv",  delimiter=",")

X = dateset[:, 0:8].astype(float)
y = dateset[:, 8]

scores = model_.evaluate(X,y)
print( model_.metrics_names[1], scores[1] * 100 )































































