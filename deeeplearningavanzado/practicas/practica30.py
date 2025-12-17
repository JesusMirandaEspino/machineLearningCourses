# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 08:36:17 2025

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


print(len(alfabeto) - seq_long)

for i in range( 0, len(alfabeto) - seq_long, 1 ):
    seq_in = alfabeto[i : i + seq_long] 
    seq_out = alfabeto[i+seq_long]
    dataX.append([ chart_to_in[char] for char in seq_in  ])
    dataY.append([ chart_to_in[seq_out] ])
    print(seq_in, "--->", seq_out)
    
    
    
    


X = pad_sequences(dataX, maxlen=seq_long, dtype='float32')
print(X)

X = np.reshape( dataX, (X.shape[0],seq_long, 1) )
print(X)



X = X/float(len(alfabeto))
print(X)

print(X.shape)

Y = to_categorical(dataY)
print(Y)


model = Sequential()
model.add(LSTM(16, input_shape=(X.shape[1],X.shape[2])))
model.add(Dense(Y.shape[1], activation='softmax'))



print(Y.shape)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
history = model.fit(X, Y, epochs=500, batch_size=len(dataX), shuffle=False)


scores = model.evaluate(X,Y)





seq_long = 1
dataX = []
dataY = []


print(len(alfabeto) - seq_long)

for i in range( 0, len(alfabeto) - seq_long, 1 ):
    seq_in = alfabeto[i : i + seq_long] 
    seq_out = alfabeto[i+seq_long]
    dataX.append([ chart_to_in[char] for char in seq_in  ])
    dataY.append([ chart_to_in[seq_out] ])
    print(seq_in, "--->", seq_out)
    
    

X = np.reshape( dataX, (len(dataX), seq_long, 1) )
print(X)



X = X/float(len(alfabeto))
print(X)

print(X.shape)

Y = to_categorical(dataY)
print(Y)

batch_size = 1

model = Sequential()
model.add(LSTM(50, batch_input_shape=(batch_size, X.shape[1],X.shape[2]), stateful=True))
model.add(Dense(Y.shape[1], activation='softmax'))



print(Y.shape)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

for i in range(300):
    model.fit(X, Y, epochs=1, batch_size=batch_size, shuffle=False)
    model.reset_states()

scores = model.evaluate(X,Y)
















































