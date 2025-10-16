# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 10:37:35 2025

@author: jesus
"""

import numpy as np # aplicacion de matematicas
import matplotlib.pyplot as plt # mostrar datos
import pandas as pd # manipular datos
#from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
#from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap


#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.svm import SVR
#from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Social_Network_Ads2.csv')

#separar
X = dataset.iloc[ :,0:2 ].values
y = dataset.iloc[ :,-1 ].values




X_train, X_test, y_train, y_test = train_test_split( X, y,  test_size=0.2, random_state=42)

sc_X = StandardScaler()


X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


#Ajustar el modelo
claficador = LogisticRegression(random_state=42)
claficador.fit(X_train,y_train)

y_pred = claficador.predict(X_test)


#matriz de confusion

cm = confusion_matrix( y_test, y_pred )


X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, claficador.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, claficador.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()







