# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 08:42:51 2025

@author: jesus
"""

import numpy as np
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers, optimizers
import pandas as pd
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
word_index = reuters.get_word_index()

print("Trainning data:", len(train_data))
print("Testing data:", len(test_data))

reversed_word_index = {v: k for k, v in word_index.items()}
' '.join([reversed_word_index.get(i - 3, '(' + str(i) + ')') for i in train_data[0]])


def vectorize(seqs, dim=10000):
    results = np.zeros((len(seqs), dim))
    for i, seq in enumerate(seqs):
        results[i, seq] = 1.
    return results


X_train = vectorize(train_data, 10000)
X_test = vectorize(test_data, 10000)

print("Valores originales:\t", train_data[0][:15])
print("Valores vectorizados:\t", X_train[0][:15])


Y_train = to_categorical(train_labels)
Y_test = to_categorical(test_labels)


X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5)


print("Longitud subconjunto de entrenamiento: ", len(X_train))
print("Longitud subconjunto de validación: ", len(X_val))
print("Longitud subconjunto de pruebas: ", len(X_test))


model = models.Sequential()
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))


opt_func = optimizers.SGD(learning_rate=0.01)

model.compile(
    optimizer=opt_func,
    loss='categorical_crossentropy',
    metrics=['accuracy', 'Precision']
)

len(X_train) / 32


history = model.fit(
    X_train,
    Y_train,
    epochs=30,
    batch_size=1,
    validation_data=(X_val, Y_val))




pd.DataFrame(history.history)[['loss', 'val_loss']].plot(figsize=(10, 6))
plt.grid(True)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot(figsize=(10, 6))
plt.grid(True)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.show()


error, accuracy, precision = model.evaluate(X_test, Y_test)


print("Error: ", error)
print("Accuracy: ", accuracy)
print("Precision: ", precision)







































