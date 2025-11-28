# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 08:51:49 2025

@author: jesus
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from functools import partial


layer = tf.keras.layers.Dense( 100, activation="relu", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(0.01) )


fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]
X_train, X_valid, X_test = X_train / 255, X_valid / 255, X_test / 255


class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


RegularizedDense = partial( tf.keras.layers.Dense, activation="relu", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(0.01) )


tf.random.set_seed(42)
tf.keras.backend.clear_session()

model = tf.keras.Sequential( [ tf.keras.layers.Flatten( input_shape=[28,28]), 
                              RegularizedDense(100),  
                              RegularizedDense(100),
                              RegularizedDense(10, activation="softmax")] )


optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))


tf.random.set_seed(42)
tf.keras.backend.clear_session()

model = tf.keras.Sequential( [ tf.keras.layers.Flatten( input_shape=[28,28]), 
                              tf.keras.layers.Dropout(rate=0.2),  
                              tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
                              tf.keras.layers.Dropout(rate=0.2),
                              tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
                              tf.keras.layers.Dropout(rate=0.2),
                              tf.keras.layers.Dense(10, activation="softmax")
                              ])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))




y_probas = np.stack( [ model( X_test,training=True )  for sample in range(100) ] )
y_proba = y_probas.mean(axis=0)


model.predict(X_test[:1]).round(3)

y_proba[0].round(3)


y_std = y_probas.std(axis=0)
y_std[0].round(3)

y_pred = y_proba.argmax(axis=1)
accuracy = (y_pred == y_test).sum() / len(y_test)




class MCDropout( tf.keras.layers.Dropout ):
    def call(self, inputs, training=False):
        return super().call(inputs, training=True)


Dropout = tf.keras.layers.Dropout
mc_model = tf.keras.Sequential([
    MCDropout(layer.rate) if isinstance(layer, Dropout) else layer
    for layer in model.layers
])
mc_model.set_weights(model.get_weights())


mc_model.summary()


dense = tf.keras.layers.Dense( 100, activation="relu", kernel_initializer="he_normal", kernel_constraint=tf.keras.constraints.max_norm(1.) )




MaxNormDense = partial(tf.keras.layers.Dense,
                       activation="relu", kernel_initializer="he_normal",
                       kernel_constraint=tf.keras.constraints.max_norm(1.))

tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    MaxNormDense(100),
    MaxNormDense(100),
    tf.keras.layers.Dense(10, activation="softmax")
])
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))















































































































