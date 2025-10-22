# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 14:28:02 2025

@author: jesus
"""

from tensorflow.keras import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

boston_housing = datasets.boston_housing

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

df_train = pd.DataFrame(np.column_stack([X_train, y_train]), columns=features)
df_train.head(10)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)


network = models.Sequential()
network.add(layers.Dense(30, activation='relu', input_shape=X_train.shape[1:]))
network.add(layers.Dense(10, activation='relu'))
network.add(layers.Dense(1))
network.summary()


scaler = RobustScaler()
X_train_prep = scaler.fit_transform(X_train)
X_val_prep = scaler.transform(X_val)
X_test_prep = scaler.transform(X_test)

network.compile(loss='mean_squared_error',
                optimizer='sgd')

history = network.fit(X_train_prep, 
                      y_train, 
                      epochs=30, 
                      validation_data=(X_val_prep, y_val))



pd.DataFrame(history.history).plot(figsize=(10, 7))
plt.grid(True)
plt.xlabel("epochs")
plt.ylabel("error")
plt.show()


test_loss = network.evaluate(X_test_prep, y_test)
print('test_mse:', test_loss)



X_new = X_test[23]
features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

df_new = pd.DataFrame([X_new], columns=features[:-1])
df_new.head()
X_new_prep = scaler.transform(df_new)
y_predict = network.predict(X_new_prep)

print("Predicción:", y_predict.round(2))
print("Valor original:", y_test[23])

network.save("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/16000_pcm_speeches/modelo_boston_housing.h5")
































