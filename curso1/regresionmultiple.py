# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 07:58:06 2025

@author: jesus
"""

#Antes, utilizábamos como variables independientes
#X_opt = X[:, [0,1,2,3,4,5]]
#siendo esto un ndarray de Python, sin embargo ahora necesitaremos declararlo como:
#X_opt = X[:, [0,1,2,3,4,5]].tolist()
#para que la función OLS siga funcionando. De hecho, en el interior de la función 
#backwardElimination donde creamos un objeto regressor_OLS, con la sintaxis
#regressor_OLS = sm.OLS(y, x).fit()
#ahora la variable independiente deberá ser modificada por
#regressor_OLS = sm.OLS(y, x.tolist()).fit()

import numpy as np # aplicacion de matematicas
import matplotlib.pyplot as plt # mostrar datos
import pandas as pd # manipular datos
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


import statsmodels.formula.api as sm
import statsmodels.api as sml

dataset = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/50_Startups.csv')

#separar
X = dataset.iloc[ :,:-1 ].values
y = dataset.iloc[ :,4 ].values


#tranformar los paises codificar datos categoricos
labelEncoder_X = LabelEncoder()

X[ :,3 ] = labelEncoder_X.fit_transform(X[ :,3 ])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],    
    remainder='passthrough'                         
)
 

X = np.array(ct.fit_transform(X),  dtype=float)

X = X[:,1:]



#separar datos
X_train, X_test, y_train, y_test = train_test_split( X, y,  test_size=0.2, random_state=42)



regresion = LinearRegression()
regresion.fit( X_train, y_train)
y_predict = regresion.predict(X_test)

X = np.append(arr = np.ones((50, 1)).astype("int64"), values = X, axis = 1)

SL = 0.05



X_opt = X[:, [0, 1, 2, 3, 4, 5]].tolist()
"""X_opt = np.array(X_opt, dtype = "float64")"""
regression_OLS = sml.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()



X_opt = X[:, [0, 1, 3, 4, 5]].tolist()
regression_OLS = sml.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]].tolist()
regression_OLS = sml.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()


X_opt = X[:, [0, 3, 5]].tolist()
regression_OLS = sml.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

X_opt = X[:, [0, 3]].tolist()
regression_OLS = sml.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()

