# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 09:26:31 2025

@author: jesus
"""

import pandas as pd
import tensorflow as ft
import numpy as np
import tensorflow as tf

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

from scikeras.wrappers import KerasClassifier, KerasRegressor

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline




dateset = np.loadtxt("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/pima-indians-diabetes.csv",  delimiter=",")

X = dateset[:, 0:8].astype(float)
y = dateset[:, 8]


# checkpoints


model = Sequential()
model.add( Dense(12, input_dim=8, activation="relu") )
model.add( Dense(8, activation="relu") )
model.add( Dense(1, activation="sigmoid") )
model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


filepath = "C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/checkpoints/weights-best.keras"
checkpoints = ModelCheckpoint(filepath, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max" )
callbacks_list = [checkpoints]


model.fit( X,y, epochs=150, batch_size=10, callbacks=callbacks_list )
scores = model.evaluate(X,y)

print( model.metrics_names[1], scores[1] * 100 )




model_ = Sequential()
model_.add( Dense(12, input_dim=8, activation="relu") )
model_.add( Dense(8, activation="relu") )
model_.add( Dense(1, activation="sigmoid") )
model_.summary()



model_.load_weights("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/checkpoints/weights-best.keras")

model_.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

dateset = np.loadtxt("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/pima-indians-diabetes.csv",  delimiter=",")

X = dateset[:, 0:8].astype(float)
y = dateset[:, 8]

scores = model_.evaluate(X,y)

print( model_.metrics_names[1], scores[1] * 100 )

































