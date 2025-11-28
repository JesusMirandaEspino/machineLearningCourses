# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 14:24:29 2025

@author: jesus
"""

from sklearn.datasets import load_sample_images
import tensorflow as tf
from functools import partial
import numpy as np
import tensorflow_datasets as tfds

images = load_sample_images()["images"]
images = tf.keras.layers.CenterCrop(height=70, width=120)(images)
images = tf.keras.layers.Rescaling(scale=1 / 255)(images)

print(images.shape)


conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7)
fmaps = conv_layer(images)

print(fmaps.shape)

conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7, padding="same")
fmaps = conv_layer(images)

print(fmaps.shape)


kernel, biases = conv_layer.get_weights()
print(kernel.shape)
print(biases.shape)

max_pool = tf.keras.layers.MaxPool2D(pool_size=2)

class DepthPool(tf.keras.layers.Layer):
    def __init__(self, pool_size=2, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        
    def call(self, inputs):
        shape = tf.shape(inputs)
        groups = shape[-1] // self.pool_size
        new_shape = tf.concat([shape[:-1], [groups, self.pool_size]], axis=0)
        return tf.reduce_max( tf.reshape(inputs, new_shape) )
    

global_avg_pool = tf.keras.layers.GlobalAvgPool2D()



global_avg_pool_2 = tf.keras.layers.Lambda( lambda X: tf.reduce_mean(X, axis=[1,2]) )


global_avg_pool_2(images)










mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = mnist
X_train_full = np.expand_dims(X_train_full, axis=-1).astype(np.float32) / 255
X_test = np.expand_dims(X_test.astype(np.float32), axis=-1) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]



DefaultConv2D = partial( tf.keras.layers.Conv2D, kernel_size=3, padding="same", activation="relu", kernel_initializer="he_normal" )

model = tf.keras.Sequential([ DefaultConv2D(filters=64, kernel_size=7, input_shape=[28,28,1]),
                             tf.keras.layers.MaxPool2D(),
                             DefaultConv2D(filters=128),
                             DefaultConv2D(filters=128),
                             tf.keras.layers.MaxPool2D(),
                             DefaultConv2D(filters=256),
                             DefaultConv2D(filters=256),
                             tf.keras.layers.MaxPool2D(),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(units=128, activation="relu", kernel_initializer="he_normal"),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(units=64, activation="relu", kernel_initializer="he_normal"),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(units=10, activation="softmax")])


model.summary()


model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))
score = model.evaluate(X_test, y_test)
X_new = X_test[:10]  # pretend we have new images
y_pred = model.predict(X_new)







DefaultConv2D = partial( tf.keras.layers.Conv2D, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal", use_bias=False )


class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [ DefaultConv2D(filters, strides=strides),
                            tf.keras.layers.BatchNormalization(),
                            self.activation,
                            DefaultConv2D(filters),
                            tf.keras.layers.BatchNormalization()]

        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [ DefaultConv2D(filters, kernel_size=1, strides=strides),
                                tf.keras.layers.BatchNormalization()]


    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
    
    
    
model = tf.keras.Sequential( [ DefaultConv2D( 64, kernel_size=7, strides=2, input_shape=[224, 224, 3] ),
                              tf.keras.layers.BatchNormalization(),
                              tf.keras.layers.Activation("relu"),
                              tf.keras.layers.MaxPool2D( pool_size=3, strides=2, padding="same" )] )

prev_filters = 64

for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
    
model.add(tf.keras.layers.GlobalAvgPool2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation="softmax"))


model.summary()



model = tf.keras.applications.ResNet50( weights="imagenet" )

images = load_sample_images()["images"]
images_resized = tf.keras.layers.Resizing( height=224, width=224, crop_to_aspect_ratio=True )(images)

inputs = tf.keras.applications.resnet50.preprocess_input(images_resized)
Y_proba = model.predict(inputs)
print(Y_proba.shape)

top_k = tf.keras.applications.resnet50.decode_predictions(Y_proba, top=3)
for image_index in range(len(images)):
    print(f"Image #{image_index}")
    for class_id, name, y_proba in top_k[image_index]:
        print(f" {class_id} - {name:12s} {y_proba:.2%}")


dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes

test_set_raw, valid_set_raw, train_set_raw = tfds.load("tf_flowers", split=[ "train[:10%]", "train[10%:25%]", "train[25%:]" ], as_supervised=True)

batch_size = 32
preprocess = tf.keras.Sequential( [ tf.keras.layers.Resizing( height=224, width=224, crop_to_aspect_ratio=True ),
                                    tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)] )

train_set = train_set_raw.map( lambda X, y: (preprocess(X), y) )
train_set = train_set.shuffle( 1000, seed=42 ).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map( lambda X, y: (preprocess(X), y) ).batch(batch_size)
test_set = test_set_raw.map( lambda X, y: (preprocess(X), y) ).batch(batch_size)


data_argumentation = tf.keras.Sequential( [ 
                                            tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
                                            tf.keras.layers.RandomRotation(factor=0.05, seed=42),
                                            tf.keras.layers.RandomContrast(factor=0.2, seed=42)] )





base_model = tf.keras.applications.xception.Xception( weights="imagenet", include_top=False )
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense( n_classes, activation="softmax" )(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=output )


for layer in base_model.layers:
    layer.trainable = False
    
    

optimizer = tf.keras.optimizers.SGD( learning_rate=0.1, momentum=0.9 )
model.compile( loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"] )
history = model.fit( train_set, validation_data=valid_set, epochs=3 )


for layer in base_model.layers[56:]:
    layer.trainable = False
    

optimizer = tf.keras.optimizers.SGD( learning_rate=0.1, momentum=0.9 )
model.compile( loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"] )
history = model.fit( train_set, validation_data=valid_set, epochs=10 )



tf.random.set_seed(42)
base_model = tf.keras.applications.xception.Xception( weights="imagenet", include_top=False )
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
class_out = tf.keras.layers.Dense( n_classes, activation="softmax" )(avg)
loc_output = tf.keras.layers.Dense(4)(avg)
model = tf.keras.Model(inputs=base_model.input, outputs=[class_out, loc_output]) 

optimizer = tf.keras.optimizers.SGD( learning_rate=0.01, momentum=0.9 )
model.compile( loss=["sparse_categorical_crossentropy", "mse"], loss_weights=[0.8, 0.2],
              optimizer=optimizer, metrics=["accuracy", "mse"])

def add_random_bounding_boxes(images, labels):
    fake_bboxes = tf.random.uniform([tf.shape(images)[0], 4])
    return images, (labels, fake_bboxes)

fake_train_set = train_set.take(5).repeat(2).map(add_random_bounding_boxes)
model.fit(fake_train_set, epochs=2)














