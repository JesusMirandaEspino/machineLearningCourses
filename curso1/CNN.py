# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 07:26:46 2025

@author: jesus
"""

# Importing the libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
import numpy as np
import keras.utils as image



train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocesamiento 
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Construcción de la CNN

# (CNN)
cnn = tf.keras.models.Sequential()

#  Capa de Convolución
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

#  Max Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Segunda capa convolucional
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Aplanamiento (Flatten)
cnn.add(tf.keras.layers.Flatten())

# Conexión Completa
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Capa de Salida
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Parte 3 - Entrenamiento de la CNN

# Compilando
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Conjunto de entrenamiento y validándola con el conjunto de prueba
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

# Hacer una predicción para una imagen individual


# Cargar una imagen para predecir
test_image = image.load_img('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/dataset/test_image/Hadelin_Dog.jpg', target_size = (64, 64))

# Convertir la imagen a un arreglo NumPy
test_image = image.img_to_array(test_image)

# Ampliar las dimensiones para que se ajuste al formato de entrada de la CNN
test_image = np.expand_dims(test_image, axis = 0)

# Realizar la predicción
result = cnn.predict(test_image)

# Ver los índices de las clases (gato o perro)
training_set.class_indices

# Asignar la predicción: 1 = perro, 0 = gato
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
