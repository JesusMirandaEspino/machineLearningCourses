# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 14:12:09 2025

@author: jesus
"""


import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras


def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = threshold * tf.abs(error) - threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn

huber_fn = create_huber(1.0)


plt.figure(figsize=(10, 5.5))

z = np.linspace(-4, 4, 200)

plt.plot(z, huber_fn(0, z), "b-", linewidth=2, label="huber($z$)")

plt.gca().axvline(x=0, color='k')
plt.axis([-4, 4, 0, 4])
plt.grid(True)
plt.xlabel("$z$")
plt.ylabel("$huber\_loss(0, z)$")
plt.legend(fontsize=14)
plt.title("Huber loss", fontsize=14)
plt.show()

boston_housing = datasets.boston_housing


(X_train, y_train), (X_test, y_test) = boston_housing.load_data()


features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

df_train = pd.DataFrame(np.column_stack([X_train, y_train]), columns=features)
df_train.head(10)


X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)


scaler = RobustScaler()

X_train_prep = scaler.fit_transform(X_train)
X_val_prep = scaler.transform(X_val)
X_test_prep = scaler.transform(X_test)


network = models.Sequential()

network.add(layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]))
network.add(layers.Dense(10, activation='relu'))
network.add(layers.Dense(1))


network.summary()

network.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mae']
               )

history = network.fit(X_train_prep, 
                      y_train, 
                      epochs=50, 
                      validation_data=(X_val_prep, y_val))


pd.DataFrame(history.history)[['mae', 'val_mae']].plot(figsize=(10, 6))
plt.grid(True)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

test_loss, test_mae = network.evaluate(X_test_prep, y_test)



print('test_mae:', test_mae)


network = models.Sequential()

network.add(layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]))
network.add(layers.Dense(10, activation='relu'))
network.add(layers.Dense(1))


network.compile(
    loss=create_huber(1.0),
    optimizer='adam',
    metrics=['mae']
               )


history = network.fit(X_train_prep, 
                      y_train, 
                      epochs=50, 
                      validation_data=(X_val_prep, y_val))


pd.DataFrame(history.history)[['mae', 'val_mae']].plot(figsize=(10, 6))
plt.grid(True)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


test_loss, test_mae = network.evaluate(X_test_prep, y_test)



print('test_mae:', test_mae)



network.save("C:/Users/jesus/IA/machinelearning/machinelearningbook/modelo_huber_loss.h5")



network2 = tf.keras.models.load_model("C:/Users/jesus/IA/machinelearning/machinelearningbook/modelo_huber_loss.h5",
                                      custom_objects={"huber_fn": create_huber(1.0)})




history = network2.fit(X_train_prep, 
                      y_train, 
                      epochs=2, 
                      validation_data=(X_val_prep, y_val))



def my_softplus(z):
    return tf.math.log(tf.exp(z) + 1.0)

# Función de inicialización personalizada
def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

# Función de regularización personalizada
def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))

# Restricción personalizada aplicada al valor de los parámetros de una capa
def my_positive_weights(weights):
    return tf.where(weights < 0., tf.zeros_like(weights), weights)

layer = keras.layers.Dense(1, activation=my_softplus,
                           kernel_initializer=my_glorot_initializer,
                           kernel_regularizer=my_l1_regularizer,
                           kernel_constraint=my_positive_weights)


network = models.Sequential()

network.add(layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]))
network.add(layers.Dense(10, activation='relu'))
network.add(layers.Dense(1, activation=my_softplus,
                         kernel_initializer=my_glorot_initializer,
                         kernel_regularizer=my_l1_regularizer,
                         kernel_constraint=my_positive_weights))




class MyL1Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))
    def get_config(self):
        return {"factor": self.factor}




network = models.Sequential()

network.add(layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]))
network.add(layers.Dense(10, activation='relu'))
network.add(layers.Dense(1, activation=my_softplus,
                         kernel_initializer=my_glorot_initializer,
                         kernel_regularizer=MyL1Regularizer(0.01),
                         kernel_constraint=my_positive_weights))





network.save("modelo_personalizado.h5")




model = keras.models.load_model(
    "modelo_personalizado.h5",
    custom_objects={
       "MyL1Regularizer": MyL1Regularizer,
       "my_positive_weights": my_positive_weights,
       "my_glorot_initializer": my_glorot_initializer,
       "my_softplus": my_softplus,
    })



network.compile(loss="mse", optimizer="sgd", metrics=[create_huber(2.0)])














