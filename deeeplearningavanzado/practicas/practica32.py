# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 10:12:09 2025

@author: jesus
"""


import pandas as pd
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Embedding, Conv1D, MaxPooling1D, LSTM
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical


texto = open( "C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/Datasets/wonderland.txt", "r", encoding="utf-8").read()
texto = texto.lower()

caracteres = sorted(list(set(texto)))
chart_to_int = dict( (c,i) for i, c in enumerate(caracteres) )
print(chart_to_int)


n_caracteres = len(texto)
n_vocabulario = len(caracteres)

print(n_caracteres)
print(n_vocabulario)

seq_long = 100
dataX = []
dataY = []
num_entradas = 1000
max_sequencias = 5



for i in range(0,  n_caracteres - seq_long, 1):
    seq_in = texto[i : i + seq_long] 
    seq_out = texto[i + seq_long] 
    dataX.append([ chart_to_int[caracter] for caracter in seq_in  ])
    dataY.append(chart_to_int[seq_out])
    print(seq_in, "--->", seq_out)
    
    
n_patrones = len(dataX)
print(n_patrones)


X = np.reshape( dataX, (n_patrones, seq_long, 1) )
print(X)



X = X/float(n_vocabulario)
print(X)

print(X.shape)

Y = to_categorical(dataY)
print(Y)



    
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

filepath = "C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/checkpoints/pesos-{epoch:02d}-{loss:.4f}.keras"
control = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode="min")
callbacks_lista = [control] 

history = model.fit(X, Y, epochs=20, batch_size=128, callbacks=callbacks_lista)


scores = model.evaluate(X,Y)


print(scores)



ruta_pesos = "C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/checkpoints/pesos-{epoch:02d}-{loss:.4f}.keras"
model.load_weights(ruta_pesos)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])



chart_to_int = dict( (c,i) for i, c in enumerate(caracteres) )

inicio = np.random.randint( 0, len(dataX) - 1 )
patron = dataX[inicio]


for i in range(1000):
    x = np.reshape(patron, (1, len(patron), 1))
    x = x / float(n_vocabulario)
    prediccion = model.predict(x, verbose=0)
    indice = np.argmax(prediccion)
    resultado = chart_to_int[prediccion]
    seq_in = [chart_to_int[valor] for valor in patron]
    sys.stdout.write(resultado)
    patron.append(indice)
    patron = patron[1: len(patron)]



model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

filepath = "C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/checkpoints/pesos-{epoch:02d}-{loss:.4f}.keras"
control = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode="min")
callbacks_lista = [control] 

history = model.fit(X, Y, epochs=20, batch_size=128, callbacks=callbacks_lista)


scores = model.evaluate(X,Y)


print(scores)



ruta_pesos = "C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/checkpoints/pesos-{epoch:02d}-{loss:.4f}.keras"
model.load_weights(ruta_pesos)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])



chart_to_int = dict( (c,i) for i, c in enumerate(caracteres) )

inicio = np.random.randint( 0, len(dataX) - 1 )
patron = dataX[inicio]




chart_to_int = dict( (c,i) for i, c in enumerate(caracteres) )

inicio = np.random.randint( 0, len(dataX) - 1 )
patron = dataX[inicio]


for i in range(1000):
    x = np.reshape(patron, (1, len(patron), 1))
    x = x / float(n_vocabulario)
    prediccion = model.predict(x, verbose=0)
    indice = np.argmax(prediccion)
    resultado = chart_to_int[prediccion]
    seq_in = [chart_to_int[valor] for valor in patron]
    sys.stdout.write(resultado)
    patron.append(indice)
    patron = patron[1: len(patron)]






































































