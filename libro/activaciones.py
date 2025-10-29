# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 15:32:08 2025

@author: jesus
"""

import tensorflow as tf


fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]
X_train, X_valid, X_test = X_train / 255, X_valid / 255, X_test / 255

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds


dense = tf.keras.layers.Dense( 50, activation="relu", kernel_initializer="he_normal" )

he_avg_init = tf.keras.initializers.VarianceScaling( scale=2., mode="fan_avg", distribution="uniform" )

dense = tf.keras.layers.Dense( 50, activation="sigmoid", kernel_initializer=he_avg_init )

leaky_relu = tf.keras.layers.LeakyReLU( negative_slope=0.2)
dense = tf.keras.layers.Dense( 50, activation="relu", kernel_initializer="he_normal" )


tf.keras.backend.clear_session()
tf.random.set_seed(42)

model = tf.keras.Sequential( [
        tf.keras.layers.Flatten( input_shape=[28,28] ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense( 300, activation="relu", kernel_initializer="he_normal" ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense( 100, activation="relu", kernel_initializer="he_normal" ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense( 10, activation="softmax" )
    ])



model.summary()


[( var.name, var.trainable ) for var in model.layers[1].variables]

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd",
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=2, validation_data=(X_valid, y_valid))



tf.keras.backend.clear_session()
tf.random.set_seed(42)

model = tf.keras.Sequential( [
        tf.keras.layers.Flatten( input_shape=[28,28] ),
        tf.keras.layers.Dense( 300, activation="relu", kernel_initializer="he_normal", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense( 100, activation="relu", kernel_initializer="he_normal", use_bias=False ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dense( 10, activation="softmax" )
    ])



model.summary()


model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd",
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=2, validation_data=(X_valid, y_valid))



optimizer = tf.keras.optimizers.SGD(clipvalue=1.0)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer)


























