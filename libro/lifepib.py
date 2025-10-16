# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 08:11:50 2025

@author: jesus
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

lifeSat = pd.read_csv("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/data-main/lifesat/lifesat.csv")

print(lifeSat.info())

X = lifeSat[['GDP per capita (USD)']]
y = lifeSat[['Life satisfaction']]

lifeSat.plot( kind='scatter', grid='True', x = 'GDP per capita (USD)', y='Life satisfaction' )
plt.axis([23_500, 62_500, 4, 9])
plt.show()


print(X)
print(y)

model = LinearRegression()

model.fit(X, y)

X_new = [[37_655.2]]


predict1 = model.predict(X_new)

print(predict1)


new_model = KNeighborsRegressor( n_neighbors=3 )

new_model.fit(X,y)

predict2 = new_model.predict(X_new)

print(predict2)