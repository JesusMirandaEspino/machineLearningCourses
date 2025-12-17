# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 09:49:40 2025

@author: jesus
"""

import pandas as pd
import tensorflow as ft
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
import math
import numpy as np

import os
import pandas as pd
import skimage

from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Embedding, Conv1D, MaxPooling1D, LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import pad_sequences
from keras.applications import mobilenet, vgg16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scikeras.wrappers import KerasClassifier, KerasRegressor

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline
from keras.preprocessing import sequence
from keras.models import Model

import gdown

# flickr_style


SEED = 1000
np.random.seed(SEED)
tf.random.set_seed(SEED)


url = "C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/Datasets/"


STYLE_NUM_LABELS = 5
# obtenemos el nombre de las primeras etiquetas seleccionadas
style_label_file = 'C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/Datasets/data/flickr_style/style_names.txt'

with open(style_label_file, encoding="utf-8") as f:
    style_labels = [line.strip() for line in f if line.strip()]
print(style_labels)


if STYLE_NUM_LABELS > 0:
    style_labels = style_labels[:STYLE_NUM_LABELS]

train_valid_frame = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/Datasets/data/flickr_style/train.txt', sep=" ", header=None)
train_valid_frame.columns = ['files','lab_idx']
train_valid_frame['labels'] = train_valid_frame['lab_idx'].map({i:j for i,j in enumerate(style_labels)})

train_valid_frame['files']  = "C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/Datasets/"+train_valid_frame['files']

train_frame, valid_frame = train_test_split(
    train_valid_frame, test_size=0.2, random_state=SEED, stratify=train_valid_frame['labels'])

test_frame = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/Datasets/data/flickr_style/test.txt', sep=" ", header=None)
test_frame.columns = ['files','lab_idx']
test_frame['labels'] = test_frame['lab_idx'].map({i:j for i,j in enumerate(style_labels)})
test_frame['files']  = "C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/cursodatasets/Datasets/"+test_frame['files']


plot_n_images = 5
fig = plt.figure(figsize=(20, 25))

np.random.seed(1000)
for i in range(0,STYLE_NUM_LABELS):
    select_frame = train_frame[train_frame['lab_idx']==i]
    print(select_frame)
    for j in range(0,plot_n_images):
        aux_index = np.random.choice(select_frame.index)
        fig_i=fig.add_subplot(plot_n_images,STYLE_NUM_LABELS,j*STYLE_NUM_LABELS+i+1)
        all_url = train_frame['files'][aux_index]
        fig_i.imshow(plt.imread(all_url))
        fig_i.set_xticks(())
        fig_i.set_yticks(())
        
    fig_i.set_xlabel('Class %s' % style_labels[i])



model = mobilenet.MobileNet(weights='imagenet')
model.summary()


train_datagen = ImageDataGenerator(preprocessing_function=mobilenet.preprocess_input)
valid_datagen = ImageDataGenerator(preprocessing_function=mobilenet.preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=mobilenet.preprocess_input)


train_iter = train_datagen.flow_from_dataframe(train_frame, x_col='files', y_col='labels', target_size=(224,224), 
                                               class_model='categorical', batch_size=32, shuffle=True)


valid_iter = train_datagen.flow_from_dataframe(valid_frame, x_col='files', y_col='labels', target_size=(224,224), 
                                               class_model='categorical', batch_size=32, shuffle=True)

test_iter = train_datagen.flow_from_dataframe(test_frame, x_col='files', y_col='labels', target_size=(224,224), 
                                               class_model='categorical', batch_size=32, shuffle=False)



test_images = next(test_iter)[0]
print(test_images)

preds = model.predict(test_images)
pred_labels = mobilenet.decode_predictions(preds, top=1 )


plot_n_images = 20
fig = plt.figure(figsize=(20, 17))

for i in range(0,5):
    for j in range(0,4):
        counter=i+4+j
        fig_i = fig.add_subplot(4,5,counter+1)
        fig_i.imshow(test_images[counter])
        fig_i.set_xticks(())
        fig_i.set_yticks(())
        fig_i.set_xlabel('%s' % style_labels[counter])
        


model = vgg16.VGG16(weights='imagenet')
model.summary()


train_datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
valid_datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)



train_iter = train_datagen.flow_from_dataframe(train_frame, x_col='files', y_col='labels', target_size=(224,224), 
                                               class_model='categorical', batch_size=32, shuffle=True)


valid_iter = train_datagen.flow_from_dataframe(valid_frame, x_col='files', y_col='labels', target_size=(224,224), 
                                               class_model='categorical', batch_size=32, shuffle=True)

test_iter = train_datagen.flow_from_dataframe(test_frame, x_col='files', y_col='labels', target_size=(224,224), 
                                               class_model='categorical', batch_size=32, shuffle=False)



test_images = next(test_iter)[0]
print(test_images)

preds = model.predict(test_images)
pred_labels = mobilenet.decode_predictions(preds, top=1 )


plot_n_images = 20
fig = plt.figure(figsize=(20, 17))

for i in range(0,5):
    for j in range(0,4):
        counter=i+4+j
        fig_i = fig.add_subplot(4,5,counter+1)
        fig_i.imshow(test_images[counter])
        fig_i.set_xticks(())
        fig_i.set_yticks(())
        fig_i.set_xlabel('%s' % style_labels[counter])
        


base_model = vgg16.VGG16(weights='imagenet')


model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
model.summary()



train_datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
valid_datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)



train_iter = train_datagen.flow_from_dataframe(train_frame, x_col='files', y_col='labels', target_size=(224,224), 
                                               class_model='categorical', batch_size=32, shuffle=True)


valid_iter = train_datagen.flow_from_dataframe(valid_frame, x_col='files', y_col='labels', target_size=(224,224), 
                                               class_model='categorical', batch_size=32, shuffle=True)

test_iter = train_datagen.flow_from_dataframe(test_frame, x_col='files', y_col='labels', target_size=(224,224), 
                                               class_model='categorical', batch_size=32, shuffle=False)

print(train_frame.shape[0])

x_fc2_train = model.predict(train_iter, steps=train_frame.shape[0]//train_iter.batch_size)
y_train = train_iter.classes


x_fc2_valid = model.predict(train_iter, steps=test_frame.shape[0]//valid_iter.batch_size)
y_valid = valid_iter.classes



x_fc2_test = model.predict(train_iter, steps=test_frame.shape[0]//test_iter.batch_size)
y_test = test_iter.classes



lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(x_fc2_train, y_train)


base_model = mobilenet.MobileNet(input_shape=(224,224,3), alpha=1, include_top=False, pooling='avg', weights='imagenet')
base_model.summary()
for layer in base_model.layers:
    layer.trainable = True

x = Dropout(0.75)(base_model.output)
x = Dense(5, activation='softmax', name='flickr_out')(x)

model = Model(base_model.input, x)



train_datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
valid_datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)



train_iter = train_datagen.flow_from_dataframe(train_frame, x_col='files', y_col='labels', target_size=(224,224), 
                                               class_model='categorical', batch_size=32, shuffle=True)


valid_iter = train_datagen.flow_from_dataframe(valid_frame, x_col='files', y_col='labels', target_size=(224,224), 
                                               class_model='categorical', batch_size=32, shuffle=True)

test_iter = train_datagen.flow_from_dataframe(test_frame, x_col='files', y_col='labels', target_size=(224,224), 
                                               class_model='categorical', batch_size=32, shuffle=False)

sgd = SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

history = model.fit(train_iter, steps_per_epoch=train_frame.shape[0]//train_iter.batch_size, epochs=10, validation_data=valid_iter,
                    validation_steps=valid_frame.shape[0]//valid_iter.batch_size)


















































































