# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 07:18:29 2025

@author: jesus
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



ann = tf.keras.models.Sequential()


ann.add(tf.keras.layers.Dense(units=6, activation='relu', kernel_initializer="uniform"))
ann.add(tf.keras.layers.Dense(units=6, activation='relu', kernel_initializer="uniform"))

ann.add(tf.keras.layers.Dense(units=1, kernel_initializer="uniform", activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


ann.summary()

ann.fit(X_train, y_train, batch_size = 32, epochs = 100)


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

cm = confusion_matrix(y_test, y_pred)
print(cm)

puntuacion = accuracy_score(y_test, y_pred)

accuracy_score(y_test, y_pred)