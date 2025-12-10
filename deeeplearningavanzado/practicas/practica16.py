# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 08:13:27 2025

@author: jesus
"""



import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
plt.imshow(X_train[4], cmap=plt.get_cmap('gray'))
plt.imshow(X_train[5], cmap=plt.get_cmap('gray'))

plt.show()


num_pixel = X_train.shape[1] * X_train.shape[2]


X_train = X_train.reshape(( X_train.shape[0], num_pixel )).astype('float32')
X_test = X_test.reshape(( X_test.shape[0], num_pixel )).astype('float32')



X_train = X_train / 255
X_test = X_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



num_classes = y_test.shape[1]

def baseline_model():
    model = Sequential()
    model.add( Dense(num_pixel, input_dim=num_pixel, activation="relu") )
    model.add( Dense(num_classes, activation="softmax") )
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    return model



model = baseline_model()

history = model.fit( X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
scores = model.evaluate(X_test, y_test)

print(scores)



(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train.reshape(( X_train.shape[0], 28, 28, 1 )).astype('float32')
X_test = X_test.reshape(( X_test.shape[0],  28, 28, 1 )).astype('float32')

print(X_train)

X_train = X_train / 255
X_test = X_test / 255



y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]



def baseline_model_():
    model = Sequential()
    model.add( Conv2D(32, (5,5),  input_shape=(28,28,1), activation='relu' ) )
    model.add( MaxPooling2D() )
    model.add( Dropout(0.2) )
    model.add( Flatten() )
    model.add( Dense(128, activation="relu") )
    model.add( Dense(10, activation="softmax") )
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    return model



model = baseline_model_()

history = model.fit( X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
scores = model.evaluate(X_test, y_test)


print(scores)


def large_model():
    model = Sequential()
    model.add( Conv2D(30, (5,5),  input_shape=(28,28,1), activation='relu' ) )
    model.add( MaxPooling2D() )
    model.add( Conv2D(15, (2,2),  activation='relu' ) )
    model.add( MaxPooling2D() )
    model.add( Dropout(0.2) )
    model.add( Flatten() )
    model.add( Dense(128, activation="relu") )
    model.add( Dense(150, activation="relu") )
    model.add( Dense(num_classes, activation="softmax") )
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    return model




model = large_model()

history = model.fit( X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
scores = model.evaluate(X_test, y_test)


print(scores)













































































