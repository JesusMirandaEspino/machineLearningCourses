# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 09:43:51 2025

@author: jesus
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap

import warnings
# Ignoramos algunos warnings que se producen por invocar el modelo sin el nombre de las características
warnings.filterwarnings('ignore', category=UserWarning, message='.*X does not have valid feature names.*')

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

df = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets/TotalFeatures-ISCXFlowMeter.csv')

X_df, y_df = remove_labels(df, 'calss')
y_df = y_df.factorize()[0]



pca = PCA(n_components=2)
df_reduced = pca.fit_transform(X_df)


df_reduced = pd.DataFrame(df_reduced, columns=["c1", "c2"])





plt.figure(figsize=(12, 6))
plt.plot(df_reduced["c1"][y_df==0], df_reduced["c2"][y_df==0], "yo", label="normal")
plt.plot(df_reduced["c1"][y_df==1], df_reduced["c2"][y_df==1], "bs", label="adware")
plt.plot(df_reduced["c1"][y_df==2], df_reduced["c2"][y_df==2], "g^", label="malware")
plt.xlabel("c1", fontsize=15)
plt.ylabel("c2", fontsize=15, rotation=0)
plt.show()


pca.explained_variance_ratio_



clf_tree_reduced = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_tree_reduced.fit(df_reduced, y_df)

def plot_decision_boundary(clf, X, y, plot_training=True, resolution=1000):
    mins = X.min(axis=0) - 1
    maxs = X.max(axis=0) + 1
    x1, x2 = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
    plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="normal")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="adware")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="malware")
        plt.axis([mins[0], maxs[0], mins[1], maxs[1]])               
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

plt.figure(figsize=(12, 6))
plot_decision_boundary(clf_tree_reduced, df_reduced.values, y_df)
plt.show()

pca = PCA(n_components=0.999)
df_reduced = pca.fit_transform(X_df)


print("Número de componentes:", pca.n_components_)

df_reduced = pd.DataFrame(df_reduced, columns=["c1", "c2", "c3", "c4", "c5", "c6"])
df_reduced["Class"] = y_df
df_reduced


train_set, val_set, test_set = train_val_test_split(df_reduced)


X_train, y_train = remove_labels(train_set, 'Class')
X_val, y_val = remove_labels(val_set, 'Class')
X_test, y_test = remove_labels(test_set, 'Class')




clf_rnd = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1)
clf_rnd.fit(X_train, y_train)



y_val_pred = clf_rnd.predict(X_val)


print("F1 score validation test:", f1_score(y_val_pred, y_val, average='weighted'))



y_test_pred = clf_rnd.predict(X_test)



print("F1 score test set:", f1_score(y_test_pred, y_test, average='weighted'))




























