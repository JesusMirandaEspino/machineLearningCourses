# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 15:41:23 2025

@author: jesus
"""

#Salary_Data


import numpy as np # aplicacion de matematicas
import matplotlib.pyplot as plt # mostrar datos
import pandas as pd # manipular datos


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Salary_Data.csv')

#separar
X = dataset.iloc[ :,:-1 ].values
y = dataset.iloc[ :,1 ].values


#separar datos
X_train, X_test, y_train, y_test = train_test_split( X, y,  test_size=0.2, random_state=42)


regresion = LinearRegression()
regresion.fit( X_train, y_train)

y_predict = regresion.predict(X_test)


plt.scatter( X_train, y_train, color="red")
plt.plot(X_train, regresion.predict(X_train), color="blue")
plt.title("Predicion Sueldos con Años de Experiencia")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo")
plt.show()


plt.scatter( X_test, y_test, color="red")
plt.plot(X_train, regresion.predict(X_train), color="blue")
plt.title("Predicion Sueldos con Años de Experiencia")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo")
plt.show()