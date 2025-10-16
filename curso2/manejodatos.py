# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 15:30:24 2025

@author: jesus
"""
import os
import pandas as pd
import arff
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import scatter_matrix


with open("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets/NSL-KDD/KDDTrain+.txt") as train_set:
    df = train_set.readlines()
    
df = pd.read_csv("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets/NSL-KDD/KDDTrain+.txt")


os.listdir("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets/NSL-KDD/")

with open('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets/NSL-KDD/KDDTrain+.arff', 'r') as train_set:
    df = arff.load(train_set)

print(df.keys())

atributos = [attr[0] for attr in df["attributes"]]

df = pd.DataFrame(df["data"], columns=atributos)


def load_kdd_dataset(data_path):
    """Lectura del conjunto de datos NSL-KDD."""
    with open(data_path, 'r') as train_set:
        dataset = arff.load(train_set)
    attributes = [attr[0] for attr in dataset["attributes"]]
    return pd.DataFrame(dataset["data"], columns=attributes)


df_orig = load_kdd_dataset('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/datasets/NSL-KDD/KDDTrain+.arff')



df = df_orig.copy()



df["protocol_type"].hist()

df.hist(bins=50, figsize=(20,15))
plt.show()


labelencoder = LabelEncoder()
df["class"] = labelencoder.fit_transform(df["class"])


df["protocol_type"] = labelencoder.fit_transform(df["protocol_type"])
df["service"] = labelencoder.fit_transform(df["service"])
df["flag"] = labelencoder.fit_transform(df["flag"])

corr_matrix = df.corr()
corr_matrix["class"].sort_values(ascending=False)

corr = df.corr()
fig, ax = plt.subplots(figsize=(8, 8))
ax.matshow(corr)
plt.xticks(range(len(corr.columns)), corr.columns);
plt.yticks(range(len(corr.columns)), corr.columns);



attributes = ["same_srv_rate", "dst_host_srv_count", "class", "dst_host_same_srv_rate"]

scatter_matrix(df[attributes], figsize=(12,8))
plt.show()
