# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 07:39:24 2025

@author: jesus
"""

import numpy as np # aplicacion de matematicas
import matplotlib as plt # mostrar datos
import pandas as pd # manipular datos
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

dataset = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Position_Salaries.csv')

#separar
X = dataset.iloc[ :,1:2 ].values
y = dataset.iloc[ :,2:3 ].values


# Escalado
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


regression = SVR( kernel="rbf" )
regression.fit(X, y)


#valor = 6.5

#y_predict = regression.predict(valor)




plt.pyplot.scatter(X, y, color="red" )
plt.pyplot.plot(X, regression.predict(X), color="blue")
plt.pyplot.title("Modelo de Regresion Lineal")
plt.pyplot.xlabel("Nivel Puesto")
plt.pyplot.ylabel("Sueldo $")
plt.pyplot.show()