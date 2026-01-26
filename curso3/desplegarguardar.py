# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 10:09:34 2025

@author: jesus
"""

from pathlib import Path
import tensorflow as tf

# extra code – load and split the MNIST dataset
mnist = tf.keras.datasets.mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = mnist
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# extra code – build & train an MNIST model (also handles image preprocessing)
tf.random.set_seed(42)
tf.keras.backend.clear_session()
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28], dtype=tf.uint8),
    tf.keras.layers.Rescaling(scale=1 / 255),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
]) 
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.SGD(learning_rate=1e-2),
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

mode_name = "my_mnist_model"
model_version = "0001"
model_path = "C:/sers\jesus/IA/machinelearning/machinelearningbook/archivos/saves/" + model_version
model.save(model_path, save_format="tf")