# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 14:39:12 2025

@author: jesus
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import add_dummy_feature
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import learning_curve

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from sklearn.metrics import mean_squared_error, mean_squared_log_error

from copy import deepcopy

np.random.seed(42)
m=100
X= 2 * np.random.rand(m,1)
y = 4 + 3 * X + np.random.rand(m,1)

plt.scatter(X, y, label='Series 1', color='blue', marker='o')

X_b = add_dummy_feature(X)
theta_best = np.linalg.inv( X_b.T @ X_b ) @ X_b.T @ y

X_new = np.array([ [0], [2] ])
X_new_b = add_dummy_feature(X_new)
y_predict = X_new_b @ theta_best
print(y_predict)

plt.plot(X_new, y_predict, "r",label='Predictions', color='blue')
plt.plot(X, y, "b.")
[...]
plt.show()

ling_reg = LinearRegression()
ling_reg.fit(X,y)
print(ling_reg.intercept_)
print(ling_reg.coef_)

ling_reg.predict(X_new)

theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)

np.linalg.pinv(X_b) @ y

eta = 0.1
n_epochs = 100
m = len(X_b)
np.random.seed(42)
theta = np.random.rand(2,1)

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients
    
    
print(theta)


n_epochs = 50
t0, t1 = 5, 50

def learning_schedule(t):
    return t0 / (t + t1)

np.random.seed(42)
theta = np.random.rand(2,1)

for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients

print(theta)


sgd_reg =  SGDRegressor( max_iter=1000, tol=1e-5, penalty=None, eta0=0.01, n_iter_no_change=100, random_state=42 )

sgd_reg.fit( X, y.ravel() )

print( sgd_reg.intercept_, sgd_reg.coef_ )


np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + 2 + np.random.rand(m, 1)
plt.scatter(X, y, label='Series 1', color='blue', marker='o')

poly_features = PolynomialFeatures( degree=2, include_bias=False )
X_poly = poly_features.fit_transform(X)

print(X[1:,])
print(X[0:,])

print(X_poly)


plt.scatter(X, y, label='Series 1', color='blue', marker='o')
plt.plot(X_poly[:,0], X_poly[:,1], "b.", color="red")
[...]
plt.show()

lin_reg = LinearRegression()
lin_reg.fit( X_poly, y )

print( lin_reg.intercept_, lin_reg.coef_ )


train_size, train_scores, valid_scores = learning_curve(  LinearRegression() , X, y, train_sizes=np.linspace(0.01, 1.0,40), 
                                                        cv=5, scoring="neg_root_mean_squared_error")

train_errors = -train_scores.mean(axis=1)
valid_erros = -valid_scores.mean(axis=1)

plt.plot( train_size, train_errors, "r-+", linewidth=2, label="train" )
plt.plot( train_size, valid_erros, "b-", linewidth=3, label="valid" )
[...]
plt.show()


polinominal_regression = make_pipeline( PolynomialFeatures( degree=10, include_bias=False ), LinearRegression() )

train_sizes, train_scores, valid_scores = learning_curve(  polinominal_regression , X, y, train_sizes=np.linspace(0.01, 1.0,40), 
                                                        cv=5, scoring="neg_root_mean_squared_error")
train_errors = -train_scores.mean(axis=1)
valid_erros = -valid_scores.mean(axis=1)

plt.plot( train_sizes, train_errors, "r-+", linewidth=2, label="train" )
plt.plot( train_sizes, valid_erros, "b-", linewidth=3, label="valid" )
[...]
plt.show()

ridge_reg = Ridge( alpha=0.1, solver="cholesky" )
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])

sdg_reg = SGDRegressor( penalty="l2", alpha=0.1 / m, tol=None, max_iter=1000, eta0=0.01, random_state=42 )
sdg_reg.fit(X, y.ravel())
sdg_reg.predict([[1.5]])

lasso_reg = Lasso( alpha=0.1 )
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])


elasctic_net = ElasticNet( alpha=0.1, l1_ratio=0.5 )
elasctic_net.fit(X, y)
elasctic_net.predict([[1.5]])

np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
X_train, y_train = X[: m // 2], y[: m // 2, 0]
X_valid, y_valid = X[m // 2 :], y[m // 2 :, 0]

preprocessing = make_pipeline(PolynomialFeatures(degree=90, include_bias=False),
                              StandardScaler())
X_train_prep = preprocessing.fit_transform(X_train)
X_valid_prep = preprocessing.transform(X_valid)
sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
n_epochs = 500
best_valid_rmse = float('inf')

for epoch in range(n_epochs):
    sgd_reg.partial_fit(X_train_prep, y_train)
    y_valid_predict = sgd_reg.predict(X_valid_prep)
    val_error = mean_squared_error(y_valid, y_valid_predict)
    if val_error < best_valid_rmse:
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)




















