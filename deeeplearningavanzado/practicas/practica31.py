# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 09:41:47 2025

@author: jesus
"""

from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

import numpy as np


alfabeto = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


chart_to_in = dict( (c,i) for i,c in enumerate(alfabeto))
int_to_char = dict( (i,c) for i,c in enumerate(alfabeto))


seq_long = 1
dataX = []
dataY = []
num_entradas = 1000
max_sequencias = 5

print(len(alfabeto) - seq_long)

for i in range( num_entradas):
    inicio = np.random.randint( len(alfabeto) - 2 )
    fin = np.random.randint( inicio, min(inicio+max_sequencias, len(alfabeto) - 1) )
    seq_in = alfabeto[inicio : fin + 1] 
    seq_out = alfabeto[ fin + 1]
    dataX.append([ chart_to_in[char] for char in seq_in  ])
    dataY.append([ chart_to_in[seq_out] ])
    print(seq_in, "--->", seq_out)
    
    

X = pad_sequences(dataX, maxlen=max_sequencias, dtype='float32')
print(X)

X = np.reshape( X, (X.shape[0],max_sequencias, 1) )
print(X)



X = X/float(len(alfabeto))
print(X)

print(X.shape)

Y = to_categorical(dataY)
print(Y)
    




model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1],1)))
model.add(Dense(Y.shape[1], activation='softmax'))



print(Y.shape)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
history = model.fit(X, Y, epochs=500, batch_size=len(dataX), shuffle=False)


scores = model.evaluate(X,Y)


print(scores)





















































































