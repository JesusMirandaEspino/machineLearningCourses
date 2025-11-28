# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 12:14:11 2025

@author: jesus
"""

import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow_hub as hub
from sklearn.datasets import load_sample_images
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)
    
train_data = np.c_[X_train, y_train]
valid_data = np.c_[X_valid, y_valid]
test_data = np.c_[X_test, y_test]
header_cols = housing.feature_names + ["MedianHouseValue"]
header = ",".join(header_cols)

norm_layer = tf.keras.layers.Normalization()
model = tf.keras.models.Sequential( [ norm_layer, tf.keras.layers.Dense(1) ] )

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=2e-3))
norm_layer.adapt(X_train)
model.fit( X_train, y_train, validation_data=(X_valid, y_valid), epochs=5 )



norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(X_train)
X_train_scaled = norm_layer(X_train)
X_valid_scaled = norm_layer(X_valid)


model = tf.keras.models.Sequential( [ tf.keras.layers.Dense(1) ] )
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=2e-3))
model.fit( X_train_scaled, y_train, validation_data=(X_valid_scaled, y_valid), epochs=5 )


final_model = tf.keras.Sequential( {norm_layer, model} )
X_new = X_test[:3]
y_pred = final_model(X_new)


dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(5)

new_dataset = dataset.map( lambda X, y: (norm_layer(X), y) )

class MyNormalization(tf.keras.layers.Layer):
    def adapt(self, X):
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        self.std_ = np.std(X, axis=0, keepdims=True)
        
    def call(self, inputs):
        eps = tf.keras.backend.epsilon()
        return (inputs - self.mean_) / (self.std_ + eps)

        


age = tf.constant([ [10.], [93.], [57.], [18.], [37.], [5.] ])
discretize_layer = tf.keras.layers.Discretization( bin_boundaries=[18., 50.] )
age_categories = discretize_layer(age)
print(age_categories)


discretize_layer = tf.keras.layers.Discretization( num_bins=3 )
discretize_layer.adapt(age)
age_categories = discretize_layer(age)
print(age_categories)


onehot_layer = tf.keras.layers.CategoryEncoding(num_tokens=3)
onehot_layer(age_categories)

two_age_categoricas = np.array( [ [1,0], [2,2], [2,0] ] )
onehot_layer(two_age_categoricas)


onehot_layer = tf.keras.layers.CategoryEncoding(num_tokens=3 + 3)
onehot_layer(two_age_categoricas + [0,3])


cities = ["Auckland", "Paris", "Paris", "San Francisco"]
str_lookup_layer = tf.keras.layers.StringLookup()
str_lookup_layer.adapt(cities)
str_lookup_layer([["Paris"], ["Auckland"], ["Auckland"], ["Montreal"]])


str_lookup_layer = tf.keras.layers.StringLookup(output_mode="one_hot")
str_lookup_layer.adapt(cities)
str_lookup_layer([["Paris"], ["Auckland"], ["Auckland"], ["Montreal"]])

str_lookup_layer = tf.keras.layers.StringLookup(num_oov_indices=5)
str_lookup_layer.adapt(cities)
str_lookup_layer([["Paris"], ["Auckland"], ["Foo"], ["Bar"], ["Baz"]])


hashing_layer = tf.keras.layers.Hashing(num_bins=10)
hashing_layer([["Paris"], ["Tokyo"], ["Auckland"], ["Montreal"]])


tf.random.set_seed(42)
embedding_layer = tf.keras.layers.Embedding( input_dim=5, output_dim=2 )
embedding_layer( np.array([2,4,2] ))

tf.random.set_seed(42)
ocean_prox = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
str_lookup_layer = tf.keras.layers.StringLookup()
str_lookup_layer.adapt(ocean_prox)
lookup_and_embed = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[], dtype=tf.string),  # WORKAROUND
    str_lookup_layer,
    tf.keras.layers.Embedding(input_dim=str_lookup_layer.vocabulary_size(),
                              output_dim=2)
])

lookup_and_embed( np.array(["<1H OCEAN", "ISLAND", "<1H OCEAN"] ) )


tf.random.set_seed(42)
np.random.seed(42)
X_train_num = np.random.rand(10_000, 8)
X_train_cat = np.random.choice(ocean_prox, size=10_000).astype(object)
y_train = np.random.rand(10_000, 1)
X_valid_num = np.random.rand(2_000, 8)
X_valid_cat = np.random.choice(ocean_prox, size=2_000).astype(object)
y_valid = np.random.rand(2_000, 1)


num_input = tf.keras.layers.Input( shape=[8], name="num" )
cat_input = tf.keras.layers.Input( shape=[], dtype=tf.string, name="cat" )
cat_embedings = lookup_and_embed(cat_input)
encoded_inputs = tf.keras.layers.concatenate([num_input, cat_embedings])
outputs = tf.keras.layers.Dense( 1 )(encoded_inputs)
model = tf.keras.models.Model(inputs=[num_input, cat_input], outputs=[outputs])
model.compile(loss="mse", optimizer="sgd")
 

train_data = ["To be", "!(to be)", "That's the question", "Be, be, be."]
text_vec_layer = tf.keras.layers.TextVectorization()
text_vec_layer.adapt(train_data)
text_vec_layer(["Be good!", "Question: be or be?"])

text_vec_layer = tf.keras.layers.TextVectorization(output_mode="tf_idf")
text_vec_layer.adapt( train_data )
text_vec_layer(["Be good!", "Question: be or be?"])


hub_layer = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim50/2")
sentence_embeddings = hub_layer( tf.constant( ["To be", "Not to be"] ) )
sentence_embeddings.numpy().round(2)





images = load_sample_images()["images"]
crop_image_layer = tf.keras.layers.CenterCrop(height=100, width=100)
cropped_images = crop_image_layer(images)


plt.imshow(images[0])
plt.axis("off")
plt.show()

plt.imshow(cropped_images[0] / 255)
plt.axis("off")
plt.show()



datasets = tfds.load(name="mnist")
mnist_train, mnist_test = datasets["train"], datasets["test"]


for batch in mnist_train.shuffle(10_000, seed=42).batch(32).prefetch(1):
    images = batch["image"]
    labels = batch["label"]


mnist_train = mnist_train.shuffle(buffer_size=10_000, seed=42).batch(32)
mnist_train = mnist_train.map( lambda items: (items["image"], items["label"]) )
mnist_train = mnist_train.prefetch(1)

train_set, valid_set, test_set = tfds.load( name="mnist", split=["train[:90%]", "train[90%:]", "test"], as_supervised=True )
train_set = train_set.shuffle(10_000, seed=42).batch(32).prefetch(1)
valid_set = valid_set.batch(32).cache()
test_set = test_set.batch(32).cache()
tf.random.set_seed(42)
model = tf.keras.Sequential( [ tf.keras.layers.Flatten(input_shape=(28,28)), 
                            tf.keras.layers.Dense(10, activation="softmax") ])
model.compile( loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"] )
model.fit(train_set, validation_data=valid_set, epochs=5)
test_loss, test_accuracy = model.evaluate(test_set)
















































