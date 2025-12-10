# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 08:15:47 2025

@author: jesus
"""

#Datasets

import pandas as pd
import tensorflow as ft
import numpy as np
from keras.models import Sequential, model_from_json
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


model_json = model.to_json()
with open("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/model.json", "w") as json_file:
    json_file.write(model_json)


model.save_weights("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/model.weights.h5")


json_file = open("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model =  model_from_json( loaded_model_json )

loaded_model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
scores = loaded_model.evaluate(X,y)


print( loaded_model.metrics_names[1], scores[1] * 100 )



























