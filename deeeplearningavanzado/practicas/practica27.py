# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 09:14:29 2025

@author: jesus
"""

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(784,))) # Primera capa oculta
model.add(Dense(32, activation='relu')) # Segunda capa oculta
model.add(Dense(10, activation='softmax')) # Salida (10 clases)