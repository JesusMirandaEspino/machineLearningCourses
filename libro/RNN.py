# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 08:10:05 2025

@author: jesus
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/curso4/Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

labelencoder_X_1 = LabelEncoder()
previo = X[:, 1]
previo2 = X[:, 2] 
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


tf.random.set_seed(42)
tf.keras.backend.clear_session()
# Inicializar la RNA
classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu", input_dim = 11))
classifier.add(Dropout(0.1))

# Añadir la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))
classifier.add(Dropout(0.1))

# Añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))

# Compilar la RNA
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

classifier.summary()


# Ajustamos la RNA al Conjunto de Entrenamiento
classifier.fit(X_train, y_train,  batch_size = 10, epochs = 100)


# Parte 3 - Evaluar el modelo y calcular predicciones finales

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)
y_pred = (y_pred>0.5)


cm = confusion_matrix(y_test, y_pred)
print((cm[0][0]+cm[1][1])/cm.sum())



tf.random.set_seed(42)
tf.keras.backend.clear_session()

def build_classifier():
  classifier = Sequential()
  classifier.add(Dense(units = 12, kernel_initializer = "uniform", activation = "relu", input_dim = 11))

  
  classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))

  
  classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
  
  classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
  return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 25, epochs = 50)
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10, n_jobs=-1, verbose = 1)

mean = accuracies.mean()
variance = accuracies.std()

print(mean)
print(variance)












#Dont run again see the results


def build_classifier2(optimizer):
  classifier = Sequential()
  classifier.add(Dense(units = 12, kernel_initializer = "uniform",  activation = "relu", input_dim = 11))
  classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))
  classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))
  classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
  return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {
    'batch_size' : [25,32],
    'epochs' : [50, 100], 
    'optimizer' : ['adam', 'rmsprop']
}

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy', 
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_parameters)
print(best_accuracy)

print("{'batch_size': 25, 'epochs': 50, 'optimizer': 'adam'}")
print("0.851375")





















tf.random.set_seed(42)
tf.keras.backend.clear_session()
def build_classifier3():
  classifier = Sequential()
  classifier.add(Dense(units = 12, kernel_initializer = "uniform", activation = "relu", input_dim = 11))
  classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))
  classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
  classifier.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ["accuracy"])
  return classifier

classifier = KerasClassifier(model = build_classifier, batch_size = 25, epochs = 100)

accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10, n_jobs=-1, verbose = 1)


classifier.fit(X_train, y_train,  batch_size = 25, epochs = 50)


y_pred  = classifier.predict(X_test)
y_pred = (y_pred>0.5)

cm = confusion_matrix(y_test, y_pred)
print((cm[0][0]+cm[1][1])/cm.sum())



tf.random.set_seed(42)
tf.keras.backend.clear_session()
ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



def create_model():

    model = ann

    return model



wrapped_model = KerasClassifier(model=create_model, epochs=10, batch_size=32, verbose=0)



accuracies = cross_val_score(estimator=wrapped_model,

                             X=X_train,

                             y=y_train,

                             cv=10)



print(f'k-fold cross-validation accuracies:\n{accuracies}\n')

print(f'k-fold cross-validation accuracy:\n{accuracies.mean()}\n')

print(f'Varianza entre accuracies y accuracy:\n{accuracies.std()*100} %\n')












