# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 09:32:52 2025

@author: jesus
"""

import numpy as np
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.linear_model import Perceptron
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


iris = load_iris(as_frame=True)
X = iris.data[['petal length (cm)', 'petal width (cm)']].values
y = ( iris.target == 0 )


per_clf = Perceptron(random_state=42)
per_clf.fit(X,y)

mlp_xlf = MLPClassifier( hidden_layer_sizes=[50,50,50],activation='relu', solver='adam', learning_rate='adaptive', max_iter=500,
random_state=42
)
mlp_xlf.fit(X,y)

x_new = [ [2,0.5], [3,1] ]
y_pred = per_clf.predict(x_new)
x_new = [ [2,0.5], [3,1] ]
y_pred = mlp_xlf.predict(x_new)

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)


mlp_reg = MLPRegressor( hidden_layer_sizes=[50,50,50], random_state=42 )
pipeline = make_pipeline(  StandardScaler(), mlp_reg )
pipeline.fit( X_train, y_train )
y_pred = pipeline.predict(X_valid)
mse = mean_squared_error(y_valid, y_pred)
rmse = np.sqrt(mse)
rmse_abs = mean_absolute_error(y_valid, y_pred)
































