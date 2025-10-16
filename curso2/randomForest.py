# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 15:19:18 2025

@author: jesus
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score
from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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



def evaluate_result(y_pred, y, y_prep_pred, y_prep, metric):
    print(metric.__name__, "WITHOUT preparation:", metric(y_pred, y, average='weighted'))
    print(metric.__name__, "WITH preparation:", metric(y_prep_pred, y_prep, average='weighted'))
    
    
    
df = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets/TotalFeatures-ISCXFlowMeter.csv')

print("Longitud del conjunto de datos:", len(df))
print("Número de características del conjunto de datos:", len(df.columns))


df["calss"].value_counts()


X = df.copy()
X['calss'] = X['calss'].factorize()[0]


corr_matrix = X.corr()
corr_matrix["calss"].sort_values(ascending=False)


corr_matrix[corr_matrix["calss"] > 0.05]


train_set, val_set, test_set = train_val_test_split(df)


X_train, y_train = remove_labels(train_set, 'calss')
X_val, y_val = remove_labels(val_set, 'calss')
X_test, y_test = remove_labels(test_set, 'calss')


scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)



X_train_scaled = DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_train_scaled.head(10)


X_train_scaled.describe()


clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X_train, y_train)


y_train_pred = clf_tree.predict(X_train)


print("F1 Score Train Set:", f1_score(y_train_pred, y_train, average='weighted'))


y_val_pred = clf_tree.predict(X_val)



print("F1 Score Validation Set:", f1_score(y_val_pred, y_val, average='weighted'))






# Modelo entrenado con el conjunto de datos sin escalar
clf_rnd = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_rnd.fit(X_train, y_train)



clf_rnd_scaled = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf_rnd_scaled.fit(X_train_scaled, y_train)


y_train_pred = clf_rnd.predict(X_train)
y_train_prep_pred = clf_rnd_scaled.predict(X_train_scaled)


evaluate_result(y_train_pred, y_train, y_train_prep_pred, y_train, f1_score)


y_pred = clf_rnd.predict(X_val)
y_prep_pred = clf_rnd_scaled.predict(X_val_scaled)




evaluate_result(y_pred, y_val, y_prep_pred, y_val, f1_score)












