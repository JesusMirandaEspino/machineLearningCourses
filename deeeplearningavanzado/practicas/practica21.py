# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 10:07:01 2025

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



look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
for i in range(5):
    print(trainX[i], trainY[i])



model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs= 200, batch_size=2, verbose=2)




trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Puntuación Entrenamiento: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Puntuación Validación: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))





trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[: , :] = np.nan
trainPredictPlot[look_back: len(trainPredict)+look_back, :] = trainPredict


# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[: , :] = np.nan
testPredictPlot[len(trainPredict) + (look_back*2) + 1: len(dataset)-1, : ] = testPredict

# plot baseline and predictions
plt.figure(figsize=(12, 8))
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()






model = Sequential()
model.add(Dense(12, input_dim=look_back, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs= 200, batch_size=2, verbose=2)





trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Puntuación Entrenamiento: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Puntuación Validación: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))


scaler = MinMaxScaler()
data = scaler.fit_transform(data)


train_size = int( len(data) *0.67 )
test_size =  len(data) - train_size


train, test = data[0:train_size, :], data[train_size:len(dataset) , :]

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX.shape

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


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])


testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

































