# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 15:16:26 2025

@author: jesus
"""

import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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


df = load_kdd_dataset("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets/NSL-KDD/KDDTrain+.arff")


train_set, val_set, test_set = train_val_test_split(df, stratify='protocol_type')
print("Longitud del Training Set:", len(train_set))
print("Longitud del Validation Set:", len(val_set))
print("Longitud del Test Set:", len(test_set))


X_train = train_set.drop("class", axis=1)
y_train = train_set["class"].copy()
X_train.loc[(X_train["src_bytes"]>400) & (X_train["src_bytes"]<800), "src_bytes"] = np.nan
X_train.loc[(X_train["dst_bytes"]>500) & (X_train["dst_bytes"]<2000), "dst_bytes"] = np.nan





class DeleteNanRows(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.dropna()
    
    
    
delete_nan = DeleteNanRows()
X_train_prep = delete_nan.fit_transform(X_train)


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, attributes):
        self.attributes = attributes
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        X_copy = X.copy()
        scale_attrs = X_copy[self.attributes]
        robust_scaler = RobustScaler()
        X_scaled = robust_scaler.fit_transform(scale_attrs)
        X_scaled = pd.DataFrame(X_scaled, columns=self.attributes, index=X_copy.index)
        for attr in self.attributes:
            X_copy[attr] = X_scaled[attr]
        return X_copy
    
    
custom_scaler = CustomScaler(["src_bytes", "dst_bytes"])
X_train_prep = custom_scaler.fit_transform(X_train_prep)


class CustomOneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._oh = OneHotEncoder(sparse=False)
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
        X_cat_oh = pd.DataFrame(X_cat_oh, 
                                columns=self._columns, 
                                index=X_copy.index)
        X_copy.drop(list(X_cat), axis=1, inplace=True)
        return X_copy.join(X_cat_oh)
    
    


num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('rbst_scaler', RobustScaler()),
    ])


X_train_num = X_train.select_dtypes(exclude=['object'])

X_train_prep = num_pipeline.fit_transform(X_train_num)
X_train_prep = pd.DataFrame(X_train_prep, columns=X_train_num.columns, index=X_train_num.index)



num_attribs = list(X_train.select_dtypes(exclude=['object']))
cat_attribs = list(X_train.select_dtypes(include=['object']))

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])


X_train_prep = full_pipeline.fit_transform(X_train)

X_train_prep = pd.DataFrame(X_train_prep, columns=list(pd.get_dummies(X_train)), index=X_train.index)




















