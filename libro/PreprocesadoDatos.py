# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 14:56:50 2025

@author: jesus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importar el dataset
url = "C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/curso4/"
origen = "ml-1m/movies.dat"
url_origen = url + origen

print(url_origen)

dataset = pd.read_csv(url_origen)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values






































