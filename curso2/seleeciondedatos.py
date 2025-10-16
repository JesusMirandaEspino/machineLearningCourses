# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 08:44:06 2025

@author: jesus
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier


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



train_set, val_set, test_set = train_val_test_split(df)


X_train, y_train = remove_labels(train_set, 'calss')
X_val, y_val = remove_labels(val_set, 'calss')
X_test, y_test = remove_labels(test_set, 'calss')




clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
clf_rnd.fit(X_train, y_train)


y_pred = clf_rnd.predict(X_val)


print("F1 score:", f1_score(y_pred, y_val, average='weighted'))

feature_importances = {name: score for name, score in zip(list(df), clf_rnd.feature_importances_)}



feature_importances_sorted = pd.Series(feature_importances).sort_values(ascending=False)
feature_importances_sorted.head(20)


columns = list(feature_importances_sorted.head(10).index)


X_train_reduced = X_train[columns].copy()
X_val_reduced = X_val[columns].copy()



clf_rnd = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
clf_rnd.fit(X_train_reduced, y_train)


y_pred = clf_rnd.predict(X_val_reduced)


print("F1 score:", f1_score(y_pred, y_val, average='weighted'))









