# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 14:36:32 2025

@author: jesus
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator as imageG
from tensorflow.keras.optimizers import RMSprop

import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

url = 'C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/curso4/Churn_Modelling.csv/dataset'





tf.random.set_seed(42)
tf.keras.backend.clear_session()

# Inicializar la CNN
classifier = Sequential()

classifier.add(Conv2D(filters = 32,kernel_size = (3, 3), 
                      input_shape = (64, 64, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))


classifier.add(Conv2D(filters = 32,kernel_size = (3, 3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))


classifier.add( Conv2D( 128, (3,3), activation='relu' ) )
classifier.add( MaxPooling2D( (2,2) ))

classifier.add( Conv2D( 128, (3,3), activation='relu' ) )
classifier.add( MaxPooling2D(2,2) )

classifier.add(Flatten())

# Paso 4 - Full Connection
classifier.add(Dense(units = 128, activation = "relu"))
classifier.add(Dense(units = 1, activation = "sigmoid"))

# Compilar la CNN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

classifier.summary()




train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)





training_dataset = train_datagen.flow_from_directory('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

testing_dataset = test_datagen.flow_from_directory('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

history = classifier.fit(training_dataset,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=testing_dataset,
                        validation_steps=2000)


val_history =  history.history

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


print(acc)
print(val_acc)
print(loss)
print(val_loss)

epochs = range( 1, len(acc)+1, 1 )
print(epochs)

plt.plot(epochs, acc, 'r--', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc') 
plt.title( 'Training and Validation accuracy' )
plt.ylabel('acc')
plt.ylabel('epochs')



plt.legend()
plt.figure()
plt.show()


plt.plot(epochs, loss, 'r--')
plt.plot(epochs, val_loss, 'b') 
plt.title( 'Training and Validation loss' )
plt.ylabel('acc')
plt.ylabel('epochs')

plt.legend()
plt.figure()
plt.show()

test_image = image.load_img('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_dataset.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'



test_image2 = image.load_img('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis = 0)
result2 = classifier.predict(test_image2)
training_dataset.class_indices
if result2[0][0] == 1:
    prediction2 = 'dog'
else:
    prediction2 = 'cat'


test_image2 = image.load_img('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/dataset/single_prediction/bob.jpeg', target_size = (64, 64))
test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis = 0)
result2 = classifier.predict(test_image2)
training_dataset.class_indices
if result2[0][0] == 1:
    prediction2 = 'dog'
    print('dog')
else:
    prediction2 = 'cat'
    print('cat')








