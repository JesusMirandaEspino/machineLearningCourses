# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 14:19:46 2025

@author: jesus
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_smalls_error = tf.abs( error ) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where( is_smalls_error, squared_loss, linear_loss )






plt.figure(figsize=(8, 3.5))
z = np.linspace(-4, 4, 200)
z_center = np.linspace(-1, 1, 200)
plt.plot(z, huber_fn(0, z), "b-", linewidth=2, label="huber($z$)")
plt.plot(z, z ** 2 / 2, "r:", linewidth=1)
plt.plot(z_center, z_center ** 2 / 2, "r", linewidth=2)
plt.plot([-1, -1], [0, huber_fn(0., -1.)], "k--")
plt.plot([1, 1], [0, huber_fn(0., 1.)], "k--")
plt.gca().axhline(y=0, color='k')
plt.gca().axvline(x=0, color='k')
plt.text(2.1, 3.5, r"$\frac{1}{2}z^2$", color="r", fontsize=15)
plt.text(3.0, 2.2, r"$|z| - \frac{1}{2}$", color="b", fontsize=15)
plt.axis([-4, 4, 0, 4])
plt.grid(True)
plt.xlabel("$z$")
plt.legend(fontsize=14)
plt.title("Huber loss", fontsize=14)
plt.show()




housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

input_shape = X_train.shape[1:]

tf.keras.utils.set_random_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation="relu", kernel_initializer="he_normal",
                          input_shape=input_shape),
    tf.keras.layers.Dense(1),
])



model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])

model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))




def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_smalls_error = tf.abs( error ) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss = threshold *  tf.abs(error) - threshold ** 2 / 2 
        return tf.where( is_smalls_error, squared_loss, linear_loss )
    return huber_fn




model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=["mae"])

model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))


model.save("my_model_with_a_custom_loss_threshold_2.keras")


class HubberLoss(tf.keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
        
    def call( self, y_true, y_pred ):
        error = y_true - y_pred
        is_smalls_error = tf.abs( error ) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold *  tf.abs(error) - self.threshold ** 2 / 2 
        return tf.where( is_smalls_error, squared_loss, linear_loss )

    def get_config(self):
        base_config = super().get_config()
        return { **base_config, "threshold: ":self.threshold }



tf.keras.utils.set_random_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation="relu", kernel_initializer="he_normal",
                          input_shape=input_shape),
    tf.keras.layers.Dense(1),
])

model.compile(loss=HubberLoss(2.), optimizer="nadam", metrics=["mae"])

model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))



model.save("my_model_with_a_custom_loss_class.keras") 


def my_softplus(z):
    return tf.math.log( 1.0 + tf.exp(z) )


def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt( 2. / shape[0] + shape[1] )
    return tf.random.normal( shape, stddev=stddev, dtype=dtype )


def my_l1_regularizer( weights ):
    return tf.reduce_sum( tf.abs( 0.01 * weights ) )


def my_positive_weights( weights ):
    return tf.where( weights < 0., tf.zeros_like( weights ), weights )


layer = tf.keras.layers.Dense( 1, activation=my_softplus, 
                                  kernel_initializer=my_glorot_initializer,
                                  kernel_regularizer=my_l1_regularizer,
                                  kernel_constraint=my_positive_weights )



class MyL1Regularizer( tf.keras.regularizers.Regularizers ):
    def __init_(self, factor):
        self.factor = factor
        
    def __call__(self, weights):
        return tf.reduce_sum( tf.abs( self.factor * weights ) )
    
    def get_config(self):
        return { "factor ": self.factor }






tf.keras.utils.set_random_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(30, activation="relu", kernel_initializer="he_normal",
                          input_shape=input_shape),
    tf.keras.layers.Dense(1),
])


model.compile( loss="mse", optimizer="adam", metrics=[create_huber(2.0)] )

model.fit(X_train_scaled, y_train, epochs=2)
presicion = tf.keras.metrics.Precision()

presicion([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1])

presicion([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0])

print(presicion.result())

print(presicion.variables)

presicion.reset_state


class HuberMetric( tf.keras.metrics.Metric ):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
    
    
    def update_state( self, y_true, x_pred, sample_weight=None ):
        sample_metrics = self.huber_fn(y_true, x_pred )
        self.total.assign_add( tf.reduce_sum( sample_metrics ) )
        self.count.assign_add( tf.cast( tf.size(y_true), tf.float32 ) )
        
    def result(self):
        return self.total / self.count
    
    
    def get_config(self):
        base_config = super().get_config()
        return { **base_config, "threshold": self.threshold }
    
    
    
    
    
    
    
    
class HuberMetricA(tf.keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)  # handles base args (e.g., dtype)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        sample_metrics = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(sample_metrics))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
    


exponential_layer = tf.keras.layers.Lambda( lambda x: tf.exp(x) )
    
    
    
    
class MyDense( tf.keras.layers.Layer ):
    def __init__( self, units, activation=None, **kwargs ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        
        
    def build( self, batch_input_shape ):
        self.kernel = self.add_weight( name="kernel", shape=[batch_input_shape[-1], self.units], initializer="glorot_normal" )
        self.bias = self.add_weight( name="bias", shape=[self.units], initializer="zeros" )
        
    def call(self, X):
        return self.activation( X @ self.kernel + self.bias )
    
    def get_config( self ):
        base_config = super().get_config()
        return { **base_config, "units":self.units, "activation":tf.keras.activations.serialize( self.activation ) }
    
    
    
    
class MyDenseA(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=[batch_input_shape[-1], self.units],
            initializer="he_normal")
        self.bias = self.add_weight(
            name="bias", shape=[self.units], initializer="zeros")

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": tf.keras.activations.serialize(self.activation)}
    
    
    
    
    
    
    
    
tf.keras.utils.set_random_seed(42)
model = tf.keras.Sequential([
    MyDense(30, activation="relu", input_shape=input_shape),
    MyDense(1)
])
model.compile(loss="mse", optimizer="nadam")
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
model.evaluate(X_test_scaled, y_test)
model.save("my_model_with_a_custom_layer.keras")
    
    
    
    
class MyMultiLayer( tf.keras.layers.Layer ):
    def call( self, X ):
        X1, X2 = X
        return X1 + X2, X1 * X2, X1 / X2
    
    
class MyMultiLayer2(tf.keras.layers.Layer):
    def call(self, X):
        X1, X2 = X
        print("X1.shape: ", X1.shape ," X2.shape: ", X2.shape)  # extra code
        return X1 + X2, X1 * X2, X1 / X2
    
    
    
    
    
    
    
    
inputs1 = tf.keras.layers.Input(shape=[2])
inputs2 = tf.keras.layers.Input(shape=[2])
MyMultiLayer()((inputs1, inputs2))
    
    
    
    
    
    
    
    
class MyGaussianNoise( tf.keras.layers.Layer ):
    def __init__( self, stddev, **kwargs ):
        super().__init__(**kwargs)
        self.stddev = stddev
        
    def call( self, X, training=False ):
        if training:
            noise = tf.random.normal( tf.shape(X), stddev=self.stddev )
            return X + noise
        else:
            return X
        
        
class MyGaussianNoiseA(tf.keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, X, training=None):
        if training:
            noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
            return X + noise
        else:
            return X
    
    
tf.keras.utils.set_random_seed(42)
model = tf.keras.Sequential([
    MyGaussianNoise(stddev=1.0, input_shape=input_shape),
    tf.keras.layers.Dense(30, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer="nadam")
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
model.evaluate(X_test_scaled, y_test)
    
    
    
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tf.keras.layers.layer.Dense(n_neurons, activation="relu", kernel_initializer="he_normal") 
                       for _ in range(n_layers)]
                
        def call(self, inputs):
            Z = inputs
            for layer in self.hidden:
                Z = layer(Z)
            return inputs + Z
        
        
        
        
class ResidualBlockA(tf.keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu",
                                             kernel_initializer="he_normal")
                       for _ in range(n_layers)]

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z  
        
        
        
class ResidualRegressor(tf.keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = tf.keras.layers.Dense( 30, activation="relu", kernel_initializer="he_normal" )
        
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = tf.keras.layers.Dense(output_dim)
            
        
        def call(self, inputs):
            Z = inputs
            for layer in self.hidden:
                Z = layer(Z)
            Z = self.block2()
            return self.out(Z)
    
    
    
    
    
class ReconstructingRegressor( tf.keras.Model ):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [ tf.keras.layers.Dense( 30, activation="relu", kernel_initializer="he_normal" ) 
                       for _ in range(5)]

        self.out = tf.keras.layers.Dense(output_dim)
        self.reconstruction_mean = tf.keras.metrics.Mean( name="reconstruction_error" )
        
        
    def build( self, batch_input_shape ):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = tf.keras.layers.Dense( n_inputs )
        
    def call( self, inputs, training=False ):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean( tf.square( reconstruction - inputs ) )
        self.add_loss( 0.05 * recon_loss )
        return self.out(Z)
    
   
class ReconstructingRegressorA(tf.keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [tf.keras.layers.Dense(30, activation="relu",
                                             kernel_initializer="he_normal")
                       for _ in range(5)]
        self.out = tf.keras.layers.Dense(output_dim)

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = tf.keras.layers.Dense(n_inputs)

    def call(self, inputs, training=None):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)
        return self.out(Z)
    
    
    
tf.keras.utils.set_random_seed(42)
model = ReconstructingRegressor(1)
model.compile(loss="mse", optimizer="nadam")
history = model.fit(X_train_scaled, y_train, epochs=5)
y_pred = model.predict(X_test_scaled)
    
    
def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2

w1, w2 = 5,3
eps = 1e-6
( f(w1 + eps, w2) - f(w1, w2)) / eps
( f(w1, w2 + eps) - f(w1, w2)) / eps
    

w1, w2 = tf.Variable(5.),tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(w1, w2)
    
gradients = tape.gradient(z,[w1,w2])


    
with tf.GradientTape(persistent=True) as tape:
    z = f(w1, w2)
    
dz_w1 = tape.gradient(z,w1)    
dz_w2 = tape.gradient(z,w2)       
    
    
c1, c2 = tf.constant(5.),tf.constant(3.) 
with tf.GradientTape(persistent=True) as tape:
    tape.watch(c1)
    tape.watch(c2)
    z = f(c1, c2)
    
gradients_c = tape.gradient(z, [c1,c2])
    
def f(w1, w2):
    return 3 * w1 ** 2 + tf.stop_gradient(2 * w1 * w2) 
    
with tf.GradientTape() as tape:
    z = f(w1, w2)
    
gradients = tape.gradient(z,[w1,w2])
    
    
def my_softplus(z):
    return tf.math.log( 1 + tf.exp( -tf.abs(z) ) ) + tf.maximum(0., z)
    
    
    
@tf.custom_gradient
def my_softplusA(z):
    def my_softplus_gradient(grads):
        return grads * ( 1 - 1 / (1 + tf.exp(z)) )
    result = tf.math.log( 1 + tf.exp( -tf.abs(z) ) ) + tf.maximum(0., z)    
    return result, my_softplus_gradient
    
l2_reg = tf.keras.regularizers.l2(0.05)
model = tf.keras.Sequential( [ tf.keras.layers.Dense( 30, activation="relu", 
                                                     kernel_initializer="he_normal",  kernel_regularizer=l2_reg),
                              tf.keras.layers.Dense(1,kernel_regularizer=l2_reg)] )
    
def random_batch( X, y, batch_size=32 ):
    idx = np.random.randit( len(X), size=batch_size )
    return X[idx], y[idx]


def print_status_bar(step, total, loss, metrics=None):
    metrics = " - ".join([f"{m.name}: {m.result}:.4f"
                          for m in [loss] + (metrics or []) ])
    end = "" if step < total else "\n"
    print( f"\r{step}/{total} - " + metrics, end=end )    
    
    
    
n_epochs = 5
batch_size = 32
n_steps = len(X_train)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_fn = tf.keras.losses.MeanSquaredError()
mean_loss = tf.keras.metrics.Mean()
metrics = [tf.keras.metrics.MeanAbsoluteError()]   
    

    
def cube(x):
    return x ** 3

tf_cube = tf.function(cube)
    
@tf.function    
def cube2(x):
    return x ** 3



    
    
    
    
    
    
    
    
    










