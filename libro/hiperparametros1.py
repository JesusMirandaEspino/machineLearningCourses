# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 08:36:43 2025

@author: jesus
"""

import tensorflow as tf
from tensorflow import keras
import kerastuner as kt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()


plt.figure(figsize=(20, 4))
for index, e in enumerate(img_train[:8]):
    plt.subplot(1, 8, index+1)
    plt.imshow(e, cmap=plt.cm.gray)
    plt.title('Ejemplo: ' + str(index+1))
plt.show()

img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0


def model_builder(hp):
    
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    
    hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)
    model.add(keras.layers.Dense(units = hp_units, activation='relu'))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate = hp_learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    
    return model

tuner = kt.Hyperband(
    model_builder,
    objective = 'val_accuracy', 
    max_epochs = 10,
    factor = 3,
    directory = 'test_dir',
    project_name = 'hp_tuning'
)

tuner.search(
    img_train, 
    label_train, 
    epochs = 10, 
    validation_data = (img_test, label_test)
)


best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


print("Units:", best_hps.get('units'))
print("Learning rate:", best_hps.get('learning_rate'))


model = tuner.hypermodel.build(best_hps)
history = model.fit(
    img_train,
    label_train,
    epochs = 10,
    validation_data = (img_test, label_test)
)




pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot(figsize=(10, 7))
plt.grid(True)
plt.xlabel("epochs")
plt.show()
































