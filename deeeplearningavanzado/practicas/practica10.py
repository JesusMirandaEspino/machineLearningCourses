# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 09:59:02 2025

@author: jesus
"""

import pandas as pd
import tensorflow as ft
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 

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
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"],)


history = model.fit( X,y, epochs=150, batch_size=10,  validation_split=0.33)
print(history.history.keys())


plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy del modelo")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Entrenamiento", "Validación"], loc="upper left")
plt.show()




plt.plot( history.history["loss"] )
plt.plot( history.history["val_loss"] )
plt.title("Loss del modelo")
plt.xlabel("Loss")
plt.ylabel("Epoch")
plt.legend(["Entrenamiento", "Validicion"], loc="upper left")
plt.show()




























