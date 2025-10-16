# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 07:38:54 2025

@author: jesus
"""

import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay


def load_kdd_dataset(data_path):
    """Lectura del conjunto de datos NSL-KDD."""
    with open(data_path, 'r') as train_set:
        dataset = arff.load(train_set)
    attributes = [attr[0] for attr in dataset["attributes"]]
    return pd.DataFrame(dataset["data"], columns=attributes)


def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)


num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('rbst_scaler', RobustScaler()),
    ])


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self._oh = OneHotEncoder()
        self._columns = None
        
    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['object'])
        self._columns = pd.get_dummies(X_cat).columns
        self._oh.fit(X_cat)
        return self
        
    def transform(self, X, y=None):
        X_copy = X.copy()
        X_cat = X_copy.select_dtypes(include=['object'])
        X_num = X_copy.select_dtypes(exclude=['object'])
        X_cat_oh = self._oh.transform(X_cat)
        X_cat_oh = pd.DataFrame(X_cat_oh.toarray(), 
                                columns=self._columns, 
                                index=X_copy.index)
        X_copy.drop(list(X_cat), axis=1, inplace=True)
        return X_copy.join(X_cat_oh)
    
    
    
    
class DataFramePreparer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self._full_pipeline = None
        self._columns = None
        
    def fit(self, X, y=None):
        num_attribs = list(X.select_dtypes(exclude=['object']))
        cat_attribs = list(X.select_dtypes(include=['object']))
        self._full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", CustomOneHotEncoder(), cat_attribs),
        ])
        self._full_pipeline.fit(X)
        self._columns = pd.get_dummies(X).columns
        return self
        
    def transform(self, X, y=None):
        X_copy = X.copy()
        X_prep = self._full_pipeline.transform(X_copy)
        return pd.DataFrame(X_prep, 
                            columns=self._columns, 
                            index=X_copy.index)
    
    
df = load_kdd_dataset("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets/NSL-KDD/KDDTrain+.arff")



train_set, val_set, test_set = train_val_test_split(df)

print("Longitud del Training Set:", len(train_set))
print("Longitud del Validation Set:", len(val_set))
print("Longitud del Test Set:", len(test_set))

X_df = df.drop("class", axis=1)
y_df = df["class"].copy()


X_train = train_set.drop("class", axis=1)
y_train = train_set["class"].copy()

X_val = val_set.drop("class", axis=1)
y_val = val_set["class"].copy()


X_test = test_set.drop("class", axis=1)
y_test = test_set["class"].copy()



data_preparer = DataFramePreparer()

data_preparer.fit(X_df)


X_train_prep = data_preparer.transform(X_train)



X_val_prep = data_preparer.transform(X_val)




clf = LogisticRegression(solver="newton-cg", max_iter=1000)
clf.fit(X_train_prep, y_train)



y_pred = clf.predict(X_val_prep)





confusion_matrix(y_val, y_pred)
ConfusionMatrixDisplay.from_estimator(clf, X_val_prep, y_val, values_format='d')


print("Precisión:", precision_score(y_val, y_pred, pos_label='anomaly'))
print("Recall:", recall_score(y_val, y_pred, pos_label='anomaly'))
print("F1 score:", f1_score(y_val, y_pred, pos_label='anomaly'))
RocCurveDisplay.from_estimator(clf, X_val_prep, y_val)



PrecisionRecallDisplay.from_estimator(clf, X_val_prep, y_val)


X_test_prep = data_preparer.transform(X_test)
y_pred = clf.predict(X_test_prep)
ConfusionMatrixDisplay.from_estimator(clf, X_test_prep, y_test, values_format='d')
print("F1 score:", f1_score(y_test, y_pred, pos_label='anomaly'))
