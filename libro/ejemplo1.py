# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 07:45:28 2025

@author: jesus
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

fashon_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashon_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]



print(X_train.shape)
print(X_train.dtype)



X_train, X_valid, X_test = X_train / 255., X_valid / 255., X_test / 255.
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]




tf.random.set_seed(42)
model = tf.keras.Sequential()
model.add( tf.keras.layers.Input( shape=[28,28] ) )
model.add( tf.keras.layers.Flatten() )
model.add( tf.keras.layers.Dense( 300, activation="relu" ) )
model.add( tf.keras.layers.Dense( 100, activation="relu" ) )
model.add( tf.keras.layers.Dense( 10, activation="softmax" ) )



model.summary()
hidden1 = model.layers[1]
print(hidden1.name)



weights, biases = hidden1.get_weights()
print(weights.shape)
print(biases.shape)



model.compile( loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"] )
history = model.fit( X_train, y_train, epochs=30, validation_data=(X_valid, y_valid) )



pd.DataFrame( history.history ).plot( figsize=(8,5), xlim=[0,29], ylim=[0,1], grid=True, xlabel="Epoch", style=["r--", "r--.", "b-", "b-*"] )
plt.show()


model.evaluate(X_test, y_test)


X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))



y_pred = y_proba.argmax(axis=1)
print(y_pred)
np.array( class_names )[y_pred]

y_new = y_test[:3]


housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)



tf.random.set_seed(42)
nom_layer = tf.keras.layers.Normalization( input_shape=X_train.shape[1:] )
model = tf.keras.Sequential( [ nom_layer, tf.keras.layers.Dense(50, activation="relu"), 
                                          tf.keras.layers.Dense(50, activation="relu"),
                                          tf.keras.layers.Dense(50, activation="relu"),
                                          tf.keras.layers.Dense(1)] )
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile( loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"] )
nom_layer.adapt(X_train)

history = model.fit( X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
mse_test, rmse_test = model.evaluate(X_test, y_test)

X_new = X_test[:3]
y_pred = model.predict(X_new)


normaliacion_layer = tf.keras.layers.Normalization()
hidden_layer1 = tf.keras.layers.Dense(30, activation="relu")
hidden_layer2 = tf.keras.layers.Dense(30, activation="relu")
concat_layer = tf.keras.layers.Concatenate()
output_layer = tf.keras.layers.Dense(1)

input_ = tf.keras.layers.Input(shape=X_train.shape[1:] )
normalized = normaliacion_layer(input_)
hidden1 = hidden_layer1(normalized)
hidden2 = hidden_layer2(hidden1)
concat = concat_layer([normalized, hidden2])
output = output_layer(concat)


model = tf.keras.Model( inputs=[input_], outputs=[output] )

model.summary()


input_wide = tf.keras.layers.Input(shape=[5])
input_deep = tf.keras.layers.Input(shape=[6])

norm_layer_wide = tf.keras.layers.Normalization()
norm_layer_deep = tf.keras.layers.Normalization()


norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)


hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)



concat = tf.keras.layers.concatenate([norm_wide, hidden2])
output = tf.keras.layers.Dense(1)(concat)
model = tf.keras.Model( inputs=[input_wide, input_deep], outputs=[output] )

model.summary()



optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile( loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"] )

X_train_wide, X_train_deep = X_train[:, :5], X_train[:, 2:]
X_valid_wide, X_valid_deep = X_valid[:, :5], X_valid[:, 2:]
X_test_wide, X_test_deep = X_test[:, :5], X_test[:, 2:]
X_new_wide, X_new_deep = X_test_wide[:3], X_test_deep[:3]


norm_layer_wide.adapt(X_train_wide)
norm_layer_deep.adapt(X_train_deep)


history = model.fit( ( X_train_wide, X_train_deep ), y_train, epochs=20, 
                    validation_data=((X_valid_wide, X_valid_deep), y_valid) )
mse_test = model.evaluate( (X_test_wide, X_test_deep), y_test )
y_pred = model.predict( (X_new_wide, X_new_deep) )



tf.keras.backend.clear_session()
tf.random.set_seed(42)



input_wide = tf.keras.layers.Input(shape=[5])  # features 0 to 4
input_deep = tf.keras.layers.Input(shape=[6])  # features 2 to 7
norm_layer_wide = tf.keras.layers.Normalization()
norm_layer_deep = tf.keras.layers.Normalization()
norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)
hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)
concat = tf.keras.layers.concatenate([norm_wide, hidden2])
output = tf.keras.layers.Dense(1)(concat)
aux_output = tf.keras.layers.Dense(1)(hidden2)
model = tf.keras.Model(inputs=[input_wide, input_deep],
                       outputs=[output, aux_output])



optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss=("mse", "mse"), loss_weights=(0.9, 0.1), optimizer=optimizer,
              metrics=["RootMeanSquaredError", "RootMeanSquaredError"])


norm_layer_wide.adapt( X_train_wide )
norm_layer_deep.adapt( X_train_deep )


history = model.fit( ( X_train_wide, X_train_deep ), (y_train, y_train), epochs=20, 
                    validation_data=((X_valid_wide, X_valid_deep), (y_valid,y_valid) ))


eval_results = model.evaluate( (X_test_wide, X_test_deep), (y_test, y_test) )
weighted_sum_of_losses, main_loss, aux_loss, main_rmse, aux_rmse = eval_results


y_pred_main, y_pred_aux = model.predict( ( X_new_wide, X_new_deep ) )
print(y_pred_main)
print(y_pred_aux)

y_pred_tuple = model.predict( ( X_new_wide, X_new_deep )  )
print(y_pred_tuple)

y_pred = dict( zip(model.output_names, y_pred_tuple) )
print(y_pred)












































