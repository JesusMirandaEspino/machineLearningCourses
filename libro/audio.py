# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 08:31:27 2025

@author: jesus
"""

import os
import IPython.display as ipd
import librosa
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import librosa.display

BENJAMIN_DATA = "C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/16000_pcm_speeches/Benjamin_Netanyau"
JENS_DATA = "C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/16000_pcm_speeches/Jens_Stoltenberg"
JULIA_DATA =  "C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/16000_pcm_speeches/Julia_Gillard"
MARGARET_DATA =  "C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/16000_pcm_speeches/Magaret_Tarcher"
NELSON_DATA =  "C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/16000_pcm_speeches/Nelson_Mandela"

ipd.Audio("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/16000_pcm_speeches/Benjamin_Netanyau/22.wav")

wav, sr = librosa.load("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/16000_pcm_speeches/Benjamin_Netanyau/22.wav")

print(wav)

print(sr)


long_audio = len(wav)/sr
print("La longitud del audio en segundos es:", long_audio)

plt.plot(wav)
plt.show()

plt.plot(wav[1000:1200])
plt.show()

wav, sr = librosa.load("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/16000_pcm_speeches/Benjamin_Netanyau/22.wav", sr=None)

print("Tasa de muestreo: {} Hz".format(sr))

def parse_dataset(dataset_paths):
    X = []
    y = []
    for index, dataset in enumerate(dataset_paths):
        print("[+] Parsing {} data...".format(dataset))
        for fname in os.listdir(dataset):
            wav, sr = librosa.load(os.path.join(dataset, fname), sr=None)
            X.append(wav)
            y.append(index)
    return (X, y)



X, y = parse_dataset([BENJAMIN_DATA, JENS_DATA])
print("La longitud del conjunto de datos es: ", len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
print("Longitud del subconjunto de entrenamiento: ", len(X_train))
print("Longitud del subconjunto de pruebas: ", len(X_test))

clf = MLPClassifier(activation='logistic', hidden_layer_sizes=(10,), solver='sgd')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
f1_score(y_test, y_pred, average="binary")
confusion_matrix(y_test, y_pred)


signal = np.cos(np.arange(0, 20, 0.2))
plt.plot(signal)
plt.show()

signal = 2*np.cos(np.arange(0, 20, 0.2)*2)
plt.plot(signal)
plt.show()

cos1 = np.cos(np.arange(0, 20, 0.2))
cos2 = 2*np.cos(np.arange(0, 20, 0.2)*2)
cos3 = 8*np.cos(np.arange(0, 20, 0.2)*4)
signal = cos1 + cos2 + cos3
plt.plot(signal)
plt.show()


cos3 = 8*np.cos(np.arange(0, 20, 0.2)*4)
plt.plot(cos3)
plt.show()

fft = np.fft.fft(signal)[:50]
fft = np.abs(fft)
plt.plot(fft)
plt.show()


D = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
librosa.display.specshow(D, y_axis='linear')
plt.show()

print(D.shape)

def parse_dataset(dataset_paths):
    X = []
    y = []
    for index, dataset in enumerate(dataset_paths):
        print("[+] Parsing {} data...".format(dataset))
        for fname in os.listdir(dataset):
            wav, sr = librosa.load(os.path.join(dataset, fname), sr=None)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
            X.append(D)
            y.append(index)
    return (X, y)

X_prep, y_prep = parse_dataset([JENS_DATA, JULIA_DATA, MARGARET_DATA, NELSON_DATA])

X_prep[100].max()

X_train, X_test, y_train, y_test = train_test_split(X_prep, y_prep, test_size=0.05)

print(len(X_train))
print(len(X_test))

X_train_prep = np.array(X_train).reshape((len(X_train), 1025*32))
X_train_prep = np.array(X_train_prep).astype('float32') / 255
y_train_prep = np.array(y_train)

X_test_prep = np.array(X_test).reshape((len(X_test), 1025*32))
X_test_prep = np.array(X_test_prep).astype('float32') / 255
y_test_prep = np.array(y_test)

clf = MLPClassifier(activation='logistic', hidden_layer_sizes=(10,), solver='sgd')
clf.fit(X_train_prep, y_train_prep)

y_pred = clf.predict(X_test_prep)

f1_score(y_test, y_pred, average="weighted")































