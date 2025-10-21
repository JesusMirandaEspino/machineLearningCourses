# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 09:06:59 2025

@author: jesus
"""

from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784')
df = pd.DataFrame(mnist.data)

X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.15)
print(len(X_train))
print(len(X_test))


clf = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', solver='sgd')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


f1_score(y_test, y_pred, average="weighted")



index = 0
index_errors = []

for label, predict in zip(y_test, y_pred):
    if label != predict:
        index_errors.append(index)
    index += 1






















