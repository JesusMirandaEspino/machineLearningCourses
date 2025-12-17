# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 10:19:20 2025

@author: jesus
"""

import pandas as pd
import tensorflow as ft
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
import math

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Embedding, Conv1D, MaxPooling1D, LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import pad_sequences

from scikeras.wrappers import KerasClassifier, KerasRegressor

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline
from keras.preprocessing import sequence


from keras.datasets import imdb

max_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_words)

max_long = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_long)
X_test = sequence.pad_sequences(X_test, maxlen=max_long)

print(X_train)


embedding_len = 32

model = Sequential()
model.add(Embedding(max_words, embedding_len, input_length=max_long ))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
model.fit(X_train, y_train, epochs= 2, batch_size=64)






scores = model.evaluate(X_test, y_test, verbose=0)
print('Puntuación Entrenamiento: ', (scores[1]*100))







embedding_len = 32

model = Sequential()
model.add(Embedding(max_words, embedding_len, input_length=max_long ))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
model.fit(X_train, y_train, epochs= 2, batch_size=64)



scores = model.evaluate(X_test, y_test, verbose=0)
print('Puntuación Entrenamiento:', (scores[1]*100))




embedding_len = 32

model = Sequential()
model.add(Embedding(max_words, embedding_len, input_length=max_long ))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
model.fit(X_train, y_train, epochs= 2, batch_size=64)





embedding_len = 32

model = Sequential()
model.add(Embedding(max_words, embedding_len, input_length=max_long ))
model.add(Conv1D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
model.fit(X_train, y_train, epochs= 2, batch_size=64)





scores = model.evaluate(X_test, y_test, verbose=0)
print('Puntuación Entrenamiento:', (scores[1]*100))







































