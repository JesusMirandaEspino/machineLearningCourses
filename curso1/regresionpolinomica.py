# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 15:29:36 2025

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

dataset = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Position_Salaries.csv')

#separar
X = dataset.iloc[ :,1:2 ].values
y = dataset.iloc[ :,2 ].values


lin_reg = LinearRegression()
lin_reg.fit(X,y)

poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)


lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)


#separar datos
#X_train, X_test, y_train, y_test = train_test_split( X, y,  test_size=0.2, random_state=42)


plt.pyplot.scatter(X, y, color="red" )
plt.pyplot.plot(X, lin_reg.predict(X), color="blue")
plt.pyplot.title("Modelo de Regresion Lineal")
plt.pyplot.xlabel("Nivel Puesto")
plt.pyplot.ylabel("Sueldo $")
plt.pyplot.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape( len(X_grid), 1 )
plt.pyplot.scatter(X, y, color="red" )
plt.pyplot.plot(X, lin_reg_2.predict(X_poly), color="blue")
plt.pyplot.title("Modelo de Regresion Lineal")
plt.pyplot.xlabel("Nivel Puesto")
plt.pyplot.ylabel("Sueldo $")
plt.pyplot.show()