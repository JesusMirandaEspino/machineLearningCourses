# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 08:16:24 2025

@author: jesus
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

SEED = 1000
np.random.seed(SEED)
tf.random.set_seed(SEED)



model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(784,))) # Primera capa oculta
model.add(Dense(32, activation='relu')) # Segunda capa oculta
model.add(Dense(10, activation='softmax')) # Salida (10 clases)

model.summary()

print(model.get_config())

print(model.to_json())

np.save('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/Datasets/model_dict.npy', model.get_config())



f = open('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/Datasets/model_json.json', 'w+')
f.write(model.to_json())
f.close()



model_struct = np.load('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/Datasets/model_dict.npy', allow_pickle=True).item()
model2 = Sequential.from_config(model_struct)
model2.summary()



model3 = model_from_json( open('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/Datasets/model_json.json'), 
                         "rb").read()
model3.summary()








(x_train_valid, y_train_valid), (x_test, y_test) = mnist.load_data()

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_valid, y_train_valid, test_size=0.1, random_state=SEED, stratify=y_train_valid)

x_train = x_train.astype('float32')
x_train /= 255
x_train = x_train.reshape(x_train.shape[0],-1)
y_train = keras.utils.to_categorical(y_train, num_classes=10)
print("Dimensiones del conjunto de características de train aplanadas: {}".format(x_train.shape))
print("Dimensiones del conjunto de etiquetas de train en one hot: {}".format(y_train.shape))
print()

x_valid = x_valid.astype('float32')
x_valid /= 255
x_valid = x_valid.reshape(x_valid.shape[0],-1)
y_valid = keras.utils.to_categorical(y_valid, num_classes=10)
print("Dimensiones del conjunto de características de train aplanadas: {}".format(x_valid.shape))
print("Dimensiones del conjunto de etiquetas de train en one hot: {}".format(y_valid.shape))
print()


x_test = x_test.astype('float32')
x_test /= 255
x_test = x_test.reshape(x_test.shape[0],-1)
y_test = keras.utils.to_categorical(y_test, num_classes=10)
print("Dimensiones del conjunto de características de test aplanadas: {}".format(x_test.shape))
print("Dimensiones del conjunto de etiquetas de test en one hot: {}".format(y_test.shape))


model.compile(loss='categorical_crossentropy', optimizer='sgd')
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=10, batch_size=128)


model.get_weights()


np.save('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/Datasets/weights_np.npy', model.get_weights())

np.save('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/Datasets/weights.h5', model.get_weights())



model2.set_weights(np.load('weights_numpy.npy', allow_pickle=True))
model2.get_weights()


early_stop = EarlyStopping(monitor="val_loss")
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=10, batch_size=128, callbacks=[early_stop])


checkpoint = ModelCheckpoint(filepath='C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/Datasets/mnist_model_{epoch:02d}.keras')
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=10, batch_size=128, callbacks=[checkpoint])












































