# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 08:09:45 2025

@author: jesus
"""

import arff
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
info = df.info()


train_set, test_set = train_test_split(df, test_size=0.4, random_state=42)
set_info1 = train_set.info()
set_info2 = test_set.info()


val_set, test_set = train_test_split(test_set, test_size=0.5, random_state=42)
print("Longitud del Training Set:", len(train_set))
print("Longitud del Validation Set:", len(val_set))
print("Longitud del Test Set:", len(test_set))


train_set, test_set = train_test_split(df, test_size=0.4, random_state=42, stratify=df["protocol_type"])
print("Longitud del conjunto de datos:", len(df))


train_set, val_set, test_set = train_val_test_split(df, stratify='protocol_type')
print("Longitud del Training Set:", len(train_set))
print("Longitud del Validation Set:", len(val_set))
print("Longitud del Test Set:", len(test_set))


df["protocol_type"].hist()
plt.show()
train_set["protocol_type"].hist()
plt.show()
val_set["protocol_type"].hist()
plt.show()
test_set["protocol_type"].hist()
plt.show()