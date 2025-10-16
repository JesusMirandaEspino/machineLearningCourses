# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 14:17:42 2025

@author: jesus
"""

import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler

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


X_train = train_set.drop("class", axis=1)
y_train = train_set["class"].copy()

print("Longitud del Training Set:", len(train_set))
print("Longitud del Validation Set:", len(val_set))
print("Longitud del Test Set:", len(test_set))

X_train.loc[(X_train["src_bytes"]>400) & (X_train["src_bytes"]<800), "src_bytes"] = np.nan
X_train.loc[(X_train["dst_bytes"]>500) & (X_train["dst_bytes"]<2000), "dst_bytes"] = np.nan


X_train.isna().any()


filas_valores_nulos  = X_train[X_train.isnull().any(axis=1)]


X_train_copy = X_train.copy()


X_train_copy.dropna(subset=["src_bytes", "dst_bytes"], inplace=True)

print("El número de filas eliminadas es:", len(X_train) - len(X_train_copy))


X_train_copy = X_train.copy()

X_train_copy.drop(["src_bytes", "dst_bytes"], axis=1, inplace=True)


print("El número de atributos eliminados es:", len(list(X_train)) - len(list(X_train_copy)))

X_train_copy = X_train.copy()


media_srcbytes = X_train_copy["src_bytes"].mean()
media_dstbytes = X_train_copy["dst_bytes"].mean()

X_train_copy["src_bytes"] = X_train_copy["src_bytes"].fillna(media_srcbytes)
X_train_copy["dst_bytes"] = X_train_copy["dst_bytes"].fillna(media_dstbytes)

X_train_copy = X_train.copy()

mediana_srcbytes = X_train_copy["src_bytes"].median()
mediana_dstbytes = X_train_copy["dst_bytes"].median()

X_train_copy["src_bytes"] = X_train_copy["src_bytes"].fillna(mediana_srcbytes)
X_train_copy["dst_bytes"] = X_train_copy["dst_bytes"].fillna(mediana_dstbytes)



X_train_copy = X_train.copy()

imputer = SimpleImputer(strategy="median")
X_train_copy_num = X_train_copy.select_dtypes(exclude=['object'])
X_train_copy_num.info()



imputer.fit(X_train_copy_num)
X_train_copy_num_nonan = imputer.transform(X_train_copy_num)
X_train_copy = pd.DataFrame(X_train_copy_num_nonan, columns=X_train_copy_num.columns)

X_train_copy.head(10)


X_train = train_set.drop("class", axis=1)
y_train = train_set["class"].copy()

X_train.info()

protocol_type = X_train['protocol_type']
protocol_type_encoded, categorias = protocol_type.factorize()


for i in range(10):
    print(protocol_type.iloc[i], "=", protocol_type_encoded[i])

print(categorias)



protocol_type = X_train[['protocol_type']]

ordinal_encoder = OrdinalEncoder()
protocol_type_encoded = ordinal_encoder.fit_transform(protocol_type)


for i in range(10):
    print(protocol_type["protocol_type"].iloc[i], "=", protocol_type_encoded[i])



print(ordinal_encoder.categories_)



protocol_type = X_train[['protocol_type']]

oh_encoder = OneHotEncoder()
protocol_type_oh = oh_encoder.fit_transform(protocol_type)
protocol_type_oh

protocol_type_oh.toarray()



for i in range(10):
    print(protocol_type["protocol_type"].iloc[i], "=", protocol_type_oh.toarray()[i])
    
    
    
print(ordinal_encoder.categories_)


oh_encoder = OneHotEncoder(handle_unknown='ignore')


pd.get_dummies(X_train['protocol_type'])


X_train = train_set.drop("class", axis=1)
y_train = train_set["class"].copy()




scale_attrs = X_train[['src_bytes', 'dst_bytes']]

robust_scaler = RobustScaler()
X_train_scaled = robust_scaler.fit_transform(scale_attrs)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=['src_bytes', 'dst_bytes'])


X_train_scaled.head(10)

X_train.head(10)