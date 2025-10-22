# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 10:32:24 2025

@author: jesus
"""

from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, optimizers, layers
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()
print(train_data[0])

reversed_word_index = {v: k for k, v in word_index.items()}
' '.join([reversed_word_index.get(i - 3, "({})".format(i)) for i in train_data[0]])
print(train_labels[0])


def vectorize(seqs, dim=10000):
    results = np.zeros((len(seqs), dim))
    for i, seq in enumerate(seqs):
        results[i, seq] = 1.
    return results

a = np.zeros(15)
print("Array original:\t\t", a)
a[[0, 3, 5, 7, 9]] = 1.
print("Array modificado:\t", a)

X_train = vectorize(train_data, 10000)
X_test = vectorize(test_data, 10000)

print("Valores originales:\t", train_data[0][:15])
print("Valores vectorizados:\t", X_train[0][:15])

X_test, X_val, Y_test, Y_val = train_test_split(X_test, test_labels, test_size=0.5)
Y_train = train_labels

print("Longitud subconjunto entrenamiento: ", len(X_train))
print("Longitud subconjunto validación: ", len(X_val))
print("Longitud subconjunto pruebas: ", len(X_test))

model = models.Sequential()
model.add(layers.Dense(16, activation='tanh', input_shape=(10000,)))
model.add(layers.Dense(16, activation='tanh'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer=optimizers.SGD(learning_rate=0.001), 
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision']
)

history = model.fit(
    X_train,
    Y_train,
    epochs=40,
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

test_loss, test_acc, test_prec = model.evaluate(X_test, Y_test)

print('test_acc:', test_acc)
print('test_prec:', test_prec)

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()

reversed_word_index = {v: k for k, v in word_index.items()}

" ".join([reversed_word_index.get(i - 3, "({})".format(i)) for i in train_data[0]])

X_train_prep = []
X_test_prep = []

for review in train_data:
    X_train_prep.append(
        " ".join([reversed_word_index.get(i - 3, "") for i in review]))


for review in test_data:
    X_test_prep.append(
        " ".join([reversed_word_index.get(i - 3, "") for i in review]))

Y_train = train_labels
Y_test = test_labels

X_test_prep, X_val_prep, Y_test, Y_val = train_test_split(X_test_prep, Y_test, test_size=0.5)

print("Longitud subconjunto entrenamiento: ", len(X_train_prep))
print("Longitud subconjunto validación: ", len(X_val_prep))
print("Longitud subconjunto pruebas: ", len(X_test_prep))

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train_prep)

X_train = X_train.toarray()

pd.DataFrame(X_train, columns=[vectorizer.get_feature_names_out()])


X_test = vectorizer.transform(X_test_prep)
X_test = X_test.toarray()

X_val = vectorizer.transform(X_val_prep)
X_val = X_val.toarray()

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer=optimizers.SGD(learning_rate=0.0005), 
    loss='binary_crossentropy',
    metrics=['accuracy', 'Precision']
)

history = model.fit(
    X_train,
    Y_train,
    epochs=10,
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

test_loss, test_acc, test_prec = model.evaluate(X_test, Y_test)

print('test_acc:', test_acc)
print('test_prec:', test_prec)
























