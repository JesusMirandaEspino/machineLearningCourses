# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 08:40:26 2025

@author: jesus
"""

import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example


X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)
print(dataset)


for item in dataset:
    print(item)
    
    
X_nested = { "a": ([1,2,3], [4,5,6] ), "b":[7,8,9] }
dataset2 = tf.data.Dataset.from_tensor_slices(X_nested)

for item in dataset2:
    print(item)
    

dataset3 = tf.data.Dataset.from_tensor_slices(tf.range(10))
dataset3 = dataset3.repeat(3).batch(7)

for item in dataset3:
    print(item)
    
    
    
dataset4 = dataset3.map(lambda x: x*2 )
for item in dataset4:
    print(item)
     
 
dataset5 = dataset4.filter(lambda x: tf.reduce_sum(x) > 50 )
for item in dataset5:
   print(item)
    
    
for item in dataset5.take(2):
   print(item)  
    
    
    
dataset6 = tf.data.Dataset.range(10).repeat(2)
dataset6 = dataset6.shuffle( buffer_size=4, seed=42 ).batch(7)
for item in dataset6:
   print(item)  
    


housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)
    
    
    


def save_to_csv_files(data, name_prefix, header=None, n_parts=10):
    housing_dir = Path() / "datasets" / "housing"
    housing_dir.mkdir(parents=True, exist_ok=True)
    filename_format = "my_{}_{:02d}.csv"

    filepaths = []
    m = len(data)
    chunks = np.array_split(np.arange(m), n_parts)
    for file_idx, row_indices in enumerate(chunks):
        part_csv = housing_dir / filename_format.format(name_prefix, file_idx)
        filepaths.append(str(part_csv))
        with open(part_csv, "w") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([str(col) for col in data[row_idx]]))
                f.write("\n")
    return filepaths

train_data = np.c_[X_train, y_train]
valid_data = np.c_[X_valid, y_valid]
test_data = np.c_[X_test, y_test]
header_cols = housing.feature_names + ["MedianHouseValue"]
header = ",".join(header_cols)

train_filepaths = save_to_csv_files(train_data, "train", header, n_parts=20)
valid_filepaths = save_to_csv_files(valid_data, "valid", header, n_parts=10)
test_filepaths = save_to_csv_files(test_data, "test", header, n_parts=10)  
    
    
print("".join(open(train_filepaths[0]).readlines()[:4]))
    

filepath_dataset = tf.data.Dataset.list_files( train_filepaths, seed=42 )
n_readers = 5
    
dataset7 = filepath_dataset.interleave( lambda filepath: tf.data.TextLineDataset(filepath).skip(1), cycle_length=n_readers )
    
for line in dataset.take(5):
    print(line)
    
    
    


scaler = StandardScaler()
scaler.fit(X_train)
    
X_mean, X_std = scaler.mean_, scaler.scale_  # extra code
n_inputs = 8
    
    
def parse_csv_line(line):
    defs = [0.] * n_inputs + [tf.constant( [], dtype=tf.float32 )]
    fields = tf.io.decode_csv( line, record_defaults=defs )
    return tf.stack( fields[:-1]), tf.stack(fields[-1:])
    
    
def preprocess(line):
    x,y = parse_csv_line(line)
    return (x - X_mean) / X_std, y
    
    
preprocess(b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782')
    
    
def csv_reader_dataset( filepaths, n_readers=5, n_read_threads=None, n_parse_threads=5, shuffle_buffer_size=10_000, seed=42, batch_size=32 ):
    dataset = tf.data.Dataset.list_files( filepaths, seed=seed )
    dataset = dataset.interleave( lambda filepath: tf.data.TextLineDataset(filepath).skip(1), 
                                 cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.map( preprocess, num_parallel_calls=n_parse_threads )
    dataset = dataset.shuffle( shuffle_buffer_size, seed=seed )
    return dataset.batch( batch_size ).prefetch( 1 )

example_set = csv_reader_dataset(train_filepaths, batch_size=3)
for X_batch, y_batch in example_set.take(2):
    print("X =", X_batch)
    print("y =", y_batch)
    print()
    
    
for m in dir(tf.data.Dataset):
    if not (m.startswith("_") or m.endswith("_")):
        func = getattr(tf.data.Dataset, m)
        if hasattr(func, "__doc__"):
            print("● {:21s}{}".format(m + "()", func.__doc__.split("\n")[0]))
    
    
train_set = csv_reader_dataset( train_filepaths )
valid_set = csv_reader_dataset(valid_filepaths)
test_set = csv_reader_dataset( test_filepaths )
    
    
    
tf.keras.backend.clear_session()
tf.random.set_seed(42)  
    
    
model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation="relu", kernel_initializer="he_normal",
                          input_shape=X_train.shape[1:]),
    tf.keras.layers.Dense(1),
])
model.compile(loss="mse", optimizer="sgd")
model.fit(train_set, validation_data=valid_set, epochs=5)
    
    
    
test_mse = model.evaluate(test_set)
new_set = test_set.take(3)
y_pred = model.predict(new_set)


    
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_fn = tf.keras.losses.MeanSquaredError()

n_epochs = 5
for epoch in range(n_epochs):
    for X_batch, y_batch in train_set:
        # extra code – perform one Gradient Descent step
        #              as explained in Chapter 12
        print("\rEpoch {}/{}".format(epoch + 1, n_epochs), end="")
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    
    
@tf.function
def train_one_epoch( model, optimizer, loss_fn, train_set ):
    for X_batch, y_batch in train_set:
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean( loss_fn( y_batch, y_pred ) )
            loss = tf.add_n( [main_loss] + model.losses )
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients( zip( gradients, model.trainable_variables ) )
        
        
optimizer = tf.keras.optimizers.SGD( learning_rate=0.01 )
loss_fn = tf.keras.losses.MeanSquaredError()
for epoch in range(n_epochs):
    print("\rEpoch {}/{}".format( epoch + 1, n_epochs), end="" )
    train_one_epoch(model, optimizer, loss_fn, train_set)
    
    
    
with tf.io.TFRecordWriter("my_data.tfrecord") as f:
    f.write(b"This is the first record")
    f.write(b"And this is the second record")
    
filepaths = ["my_data.tfrecord"]
datasetA = tf.data.TFRecordDataset(filepaths)
for item in datasetA:
    print(item)
    
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter( "my_compressed.tfrecord", options ) as f:
    f.write("Compress, compress, compress!" )
datasetB = tf.data.TFRecordDataset( ["my_compressed.tfrecord"], compression_type="GZIP" )
    
    

person_example = Example(
    features=Features(
        feature={
            "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
            "id": Feature(int64_list=Int64List(value=[123])),
            "emails": Feature(bytes_list=BytesList(value=[b"a@b.com",
                                                          b"c@d.com"]))
        }))
    


with tf.io.TFRecordWriter("my_contacts.tfrecord") as f:
    for _ in range(5):
        f.write(person_example.SerializableToString())
        

feature_description = {
        "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
        "email": tf.io.VarLenFeature(tf.string)
    }        
        
        
def parse(serialized_example):
    return tf.io.parse_sequence_example(serialized_example, feature_description)


datasetC = tf.data.TFRecordDataset( ["my_contacts.tfrecord"] ).map(parse)
for parsed_example in datasetC:
    print(parsed_example)


        

datasetD = tf.data.TFRecordDataset( ["my_contacts.tfrecord"] ).batch(2).map(parse)
for parsed_example in datasetD:
    print(parsed_example)

     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        