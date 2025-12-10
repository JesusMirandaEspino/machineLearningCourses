# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 10:12:46 2025

@author: jesus
"""

import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.constraints import MaxNorm
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()


for i in range(0,9):
    plt.subplot(330+1+i)
    plt.imshow( X_train[i] )
plt.show()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


X_train = X_train / 255
X_test = X_test / 255


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


num_classes = y_test.shape[1]

model = Sequential()
model.add( Conv2D(32, (3,3),  input_shape=(32,32,3), activation='relu', kernel_constraint=MaxNorm(3)) )
model.add( Dropout(0.2) )
model.add( Conv2D(32, (3,3),  padding='same', activation='relu', kernel_constraint=MaxNorm(3)) )
model.add( MaxPooling2D() )
model.add( Flatten() )
model.add( Dense(512,  activation="relu") )
model.add( Dropout(0.5) )
model.add( Dense(num_classes, activation="softmax") )


epochs = 25
lrate = 0.01
decay = lrate/epochs

sgd = SGD(learning_rate=lrate, momentum=0.9, decay=decay)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.summary()

history = model.fit( X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
scores = model.evaluate(X_test, y_test)


print(scores)












































