# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 08:27:37 2025

@author: jesus
"""

import numpy as np
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Embedding, Conv1D, MaxPooling1D
from tensorflow.keras.constraints import MaxNorm
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from keras.datasets import imdb
from keras.preprocessing import sequence

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)
X = np.concatenate( (X_train,X_test), axis=0)
y = np.concatenate( (y_train,y_test), axis=0)

print(np.unique(np.hstack(X)))


result = [ len(X) for x in X ]

print("Media", np.mean(result), "desviacion", np.std(result))


vocab = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab)
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

model = Sequential()
model.add( Embedding(input_dim=5000, output_dim=32,  input_length=500) )
model.add( Flatten() )
model.add( Dense(250,  activation="relu") )
model.add( Dense(1, activation="sigmoid") )
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

history = model.fit( X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128)




model = Sequential()
model.add( Embedding(input_dim=5000, output_dim=32,  input_length=500) )
model.add( Conv1D(32, 3,  padding="same", activation="relu") )
model.add( MaxPooling1D() )
model.add( Flatten() )
model.add( Dense(250,  activation="relu") )
model.add( Dense(1, activation="sigmoid") )
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


history = model.fit( X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128)





























































