# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 09:48:14 2025

@author: jesus
"""


import IPython
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
import kerastuner as kt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import models, layers
import kerastuner as kt

from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

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





def model_builder(hp):
    # Definición del modelo
    model = keras.Sequential()

    # Tuning del número de neuronas de las hidden layer
    hp_units = hp.Int('units', min_value=8, max_value=128, step=8)
    
    # Tuning del número de capas
    hp_layers = hp.Int('layers', min_value=1, max_value=5, step=1)
    
    # Input layer
    model.add(layers.Dense(units=128, activation='relu', input_shape=(10000,)))
    
    # Hidden layers 
    for i in range(hp_layers):
        # Añadimos la hidden layer
        model.add(layers.Dense(units=hp_units, activation='relu'))
        
    # Output layer
    model.add(layers.Dense(46, activation='softmax'))
    
    # Seleccionamos el valor optimo entre [0.01, 0.001, 0.0001]
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model




tuner = kt.Hyperband(
    model_builder,
    objective= 'val_accuracy',
    max_epochs=10,
    factor=3,
    directory='hp_dir',
    project_name='ruters_dataset'
)




class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)



tuner.search(
    X_train, 
    Y_train,
    epochs=10,
    validation_data=(X_val, Y_val),
    callbacks=[ClearTrainingOutput()]
)




best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Units:", best_hps.get('units'))
print("Layers:", best_hps.get('layers'))




model = tuner.hypermodel.build(best_hps)

history = model.fit(
    X_train,
    Y_train,
    epochs = 20,
    validation_data = (X_val, Y_val)
)


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


model.evaluate(X_test, Y_test)




model.predict(X_test).round(0)




























