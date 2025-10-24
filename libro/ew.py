# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 14:01:14 2025

@author: jesus
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/weather_Madrid.csv")
df['CET'] = pd.to_datetime(df['CET'])
mask = (df['CET'] >= '2015-1-1') & (df['CET'] <= '2015-12-31')
df_prep = df.loc[mask][["CET", "Mean TemperatureC"]]

plt.figure(figsize=(22, 11))
plt.plot(df_prep["CET"], df_prep["Mean TemperatureC"], "bo")
plt.ylabel("Temperatura", fontsize=14)
plt.xticks(['2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12', '2016-01'], fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(df_prep["CET"], df_prep["Mean TemperatureC"], "b")
plt.ylabel("Temperatura", fontsize=12)
plt.xticks(['2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12', '2016-01'], fontsize=11)
plt.yticks(fontsize=11)
plt.grid()
plt.show()

def ewma(theta, b=0.9):
    if len(theta) == 1:
        return b*0 + (1-b)*theta[-1]
    return b*ewma(theta[:-1]) + (1-b)*theta[-1]

def apply_ewma(data, b=0.9):
    v_data = []
    for i in np.arange(1, len(data) + 1):
        print("\rProcesando ejemplo: {0}".format(i), end='')
        v_data.append(ewma(data[:i], b=b))
    return v_data

v_df = apply_ewma(list(df_prep["Mean TemperatureC"]))

plt.figure(figsize=(14, 7))
plt.plot(df_prep["CET"], df_prep["Mean TemperatureC"], c="b")
plt.plot(df_prep["CET"], v_df, c="g")
plt.ylabel("Temperatura", fontsize=14)
plt.xticks(['2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12', '2016-01'])
plt.grid()
plt.show()

def ewma(theta, b=0.9):
    if len(theta) == 1:
        return b*0 + (1-b)*theta[-1]
    return b*ewma(theta[:-1]) + (1-b)*theta[-1]

def apply_ewma_bias_corr(data, b=0.9):
    v_data = []
    for i in np.arange(1, len(data) + 1):
        print("\rProcesando ejemplo: {0}".format(i), end='')
        v_data.append(ewma(data[:i], b=b) / (1-b**i))
    return v_data

v_df_corr = apply_ewma_bias_corr(list(df_prep["Mean TemperatureC"]))

plt.figure(figsize=(14, 7))
plt.plot(df_prep["CET"], df_prep["Mean TemperatureC"], c="b")
plt.plot(df_prep["CET"], v_df_corr, c="r")
plt.ylabel("Temperatura", fontsize=14)
plt.xticks(['2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12', '2016-01'])
plt.grid()
plt.show()




































































