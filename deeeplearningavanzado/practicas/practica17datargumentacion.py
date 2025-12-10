# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 08:34:46 2025

@author: jesus
"""

import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


(X_train, y_train), (X_test, y_test) = mnist.load_data()


num_pixels =  X_train.shape[1] *  X_train.shape[2]


X_train = X_train.reshape(( X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape(( X_test.shape[0],  num_pixels)).astype('float32')

print(X_train)

X_train = X_train / 255
X_test = X_test / 255



y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]


def baseline_model_():
    model = Sequential()
    model.add( Dense(num_pixels, input_dim=num_pixels, activation="relu") )
    model.add( Dense(10, activation="softmax") )
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    return model


model = baseline_model_()

history = model.fit( X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
scores = model.evaluate(X_test, y_test)


print(scores)





(X_train, y_train), (X_test, y_test) = mnist.load_data()


num_pixels =  X_train.shape[1] *  X_train.shape[2]

X_train = X_train.reshape(( X_train.shape[0], 28, 28, 1 )).astype('float32')
X_test = X_test.reshape(( X_test.shape[0],  28, 28, 1 )).astype('float32')


#datagen = ImageDataGenerator( featurewise_center=True, featurewise_std_normalization=True )


datagen = ImageDataGenerator( zca_whitening=True )
datagen.fit( X_train )

for X_batch, y_batch in datagen.flow( X_train, y_train, batch_size=9 ):
    for i in range(0,9):
        plt.subplot(330+1+i)
        plt.imshow( X_batch[i].reshape(28,28), cmap=plt.get_cmap('gray') )
    plt.show()
    break


datagen = ImageDataGenerator( rotation_range=90 )
datagen.fit( X_train )


for X_batch, y_batch in datagen.flow( X_train, y_train, batch_size=9 ):
    for i in range(0,9):
        plt.subplot(330+1+i)
        plt.imshow( X_batch[i].reshape(28,28), cmap=plt.get_cmap('gray') )
    plt.show()
    break


shift = 0.2
datagen = ImageDataGenerator( width_shift_range=shift, height_shift_range=shift )
datagen.fit( X_train )


for X_batch, y_batch in datagen.flow( X_train, y_train, batch_size=9 ):
    for i in range(0,9):
        plt.subplot(330+1+i)
        plt.imshow( X_batch[i].reshape(28,28), cmap=plt.get_cmap('gray') )
    plt.show()
    break




datagen = ImageDataGenerator( horizontal_flip=True, vertical_flip=True )
datagen.fit( X_train )

for X_batch, y_batch in datagen.flow( X_train, y_train, batch_size=9 ):
    for i in range(0,9):
        plt.subplot(330+1+i)
        plt.imshow( X_batch[i].reshape(28,28), cmap=plt.get_cmap('gray') )
    plt.show()
    break





datagen = ImageDataGenerator( horizontal_flip=True, vertical_flip=True )
datagen.fit( X_train )

os.makedirs("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/digits")

for X_batch, y_batch in datagen.flow( X_train, y_train, batch_size=9, save_to_dir="C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/digits",
                                     save_prefix="aug", save_format="png"):
    for i in range(0,9):
        plt.subplot(330+1+i)
        plt.imshow( X_batch[i].reshape(28,28), cmap=plt.get_cmap('gray') )
    plt.show()
    break



































































