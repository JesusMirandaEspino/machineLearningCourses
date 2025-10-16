# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 15:12:32 2025

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
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Position_Salaries.csv')

#separar
X = dataset.iloc[ :,1:2 ].values
y = dataset.iloc[ :,2:3 ].values




regression = RandomForestRegressor( n_estimators=300, random_state=42 )
regression.fit(X,y)

y_predict = regression.predict([[6.50]])

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape( len(X_grid), 1 )
plt.pyplot.scatter(X, y, color="red" )
plt.pyplot.plot(X, regression.predict(X), color="blue")
plt.pyplot.title("Modelo de Random Forest")
plt.pyplot.xlabel("Nivel Puesto")
plt.pyplot.ylabel("Sueldo $")
plt.pyplot.show()