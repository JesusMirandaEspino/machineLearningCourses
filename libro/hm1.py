# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 09:09:35 2025

@author: jesus
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers
# Importar el data set
dataset = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/curso4/Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Codificar datos categóricos

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#El OneHotEncoder en las nuevas versiones está OBSOLETO
#onehotencoder = OneHotEncoder(categorical_features=[1])
#X = onehotencoder.fit_transform(X).toarray()



transformer = ColumnTransformer(
    transformers=[
        ("Churn_Modelling",        # Un nombre de la transformación
         OneHotEncoder(categories='auto'), # La clase a la que transformar
         [1]            # Las columnas a transformar.
         )
    ], remainder='passthrough'
)

X = transformer.fit_transform(X)
X = X[:, 1:]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado de variables

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Parte 2 - Construir la RNA

# Importar Keras y librerías adicionales

tf.random.set_seed(42)
tf.keras.backend.clear_session()

# Inicializar la RNA
classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
classifier.add(Dense(units = 12, kernel_initializer = "uniform",  
                     activation = "relu", input_dim = 11))

# Añadir la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))

# Añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))

# Compilar la RNA
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Ajustamos la RNA al Conjunto de Entrenamiento
classifier.fit(X_train, y_train,  batch_size = 25, epochs = 50)


# Parte 3 - Evaluar el modelo y calcular predicciones finales

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)
y_pred = (y_pred>0.5)


# Predecir una nueva observación

"""Utiliza nuestro modelo de RNA para predecir si el cliente con la siguiente información abandonará el banco:

*   Geografia: Francia
*   Puntaje de crédito: 600
*   Género masculino
*   Edad: 40 años de edad
*   Tenencia: 3 años.
*   Saldo: $ 60000

*   Número de productos: 2
*   ¿Este cliente tiene una tarjeta de crédito? Sí
*   ¿Es este cliente un miembro activo? Sí
*   Salario estimado: $ 50000

Entonces, ¿deberíamos decir adiós a ese cliente?"""
new_prediction = classifier.predict(sc_X.transform(np.array([[0,0,600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Elaborar una matriz de confusión

cm = confusion_matrix(y_test, y_pred)
print((cm[0][0]+cm[1][1])/cm.sum())
