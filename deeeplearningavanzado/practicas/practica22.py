# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 09:17:37 2025

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


dataset = pd.read_csv("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/Datasets/international-airline-passengers.csv",  
                      usecols=[1], engine='python')
plt.plot(dataset)
plt.show()


data = dataset.values
data = data.astype('float32')


scaler = MinMaxScaler()
data = scaler.fit_transform(data)


train_size = int( len(data) *0.67 )
test_size =  len(data) - train_size

train, test = data[0:train_size, :], data[train_size:len(dataset) , :]


def create_dataset(dataset, look_back=1):
    dataX, dataY =[], []
    for i in range(len(dataset) - look_back-1):
        a = dataset[i: (i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)



look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
for i in range(5):
    print(trainX[i], trainY[i])



trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))



model = Sequential()
model.add(LSTM(4, input_shape=(1,look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs= 100, batch_size=2, verbose=2)


trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Puntuación Entrenamiento: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Puntuación Validación: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))





look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
for i in range(5):
    print(trainX[i], trainY[i])



trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))



model = Sequential()
model.add(LSTM(4, input_shape=(look_back,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs= 100, batch_size=2, verbose=2)


trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Puntuación Entrenamiento: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Puntuación Validación: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))




batch_size = 1
model = Sequential()
model.add(LSTM(4, input_shape=(look_back,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
for i in range(100):
    model.fit(trainX, trainY, epochs= 1, batch_size=2, shuffle=False )
    model.layers[0].reset_states()

    
    

trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Puntuación Entrenamiento: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Puntuación Validación: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
















