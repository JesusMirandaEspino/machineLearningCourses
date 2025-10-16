# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 08:27:49 2025

@author: jesus
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = load_iris(as_frame=True)
list(iris)

X = iris.data[["petal width (cm)"]].values
y = iris.target_names[iris.target] == "virginica"
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X,y)

X_new = np.linspace(0,3, 1000).reshape(-1,1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:,1] >= 0.5][0,0]

plt.plot( X_new, y_proba[:,0], "b--", linewidth=2, label="Not iris virginica proba" )
plt.plot( X_new, y_proba[:,1], "g-",  linewidth=2, label="Iris virginica proba")
plt.plot([decision_boundary, decision_boundary], [0,1], "k:",  linewidth=2, label="Desicion Boundary")
[...]
plt.show()

log_reg.predict( [ [1.7], [1.5]] )

X = iris.data[["petal length (cm)","petal width (cm)"]]
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)

softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train, y_train)

softmax_reg.predict([[5,2]])
softmax_reg.predict_proba([[5,2]]).round(2)
preresult = softmax_reg.predict([[5,2]])

result = softmax_reg.predict(X_test)























































