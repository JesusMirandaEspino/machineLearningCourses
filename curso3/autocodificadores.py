# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 07:46:32 2025

@author: jesus
"""

import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.manifold import TSNE


fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]


conv_encoder = tf.keras.Sequential([
        tf.keras.layers.Reshape([28,28,1]),
        tf.keras.layers.Conv2D( 16,3, padding="same", activation="relu" ),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAvgPool2D()
    ])


conv_decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(3*3*16),
        tf.keras.layers.Reshape((3,3,16)),
        tf.keras.layers.Conv2DTranspose(32,3, strides=2, activation="relu"),
        tf.keras.layers.Conv2DTranspose(16,3, strides=2, padding="same", activation="relu"),
        tf.keras.layers.Conv2DTranspose(1,3, strides=2, padding="same"),
        tf.keras.layers.Reshape([28,28])
    ])


conv_ae = tf.keras.Sequential([ conv_encoder, conv_decoder ])


dropout_encoder = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(30, activation="relu")
    ])

dropout_decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(28 * 28),
        tf.keras.layers.Reshape([28,28])
    ])


dropout_ae = tf.keras.Sequential([ dropout_encoder, dropout_decoder ])


sparse_l1_encoder = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(300, activation="sigmoid"),
        tf.keras.layers.ActivityRegularization(l1=1e-4)
    ])



sparse_l1_decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(28 * 28),
        tf.keras.layers.Reshape([28,28])
    ])



sparse_l1_ae = tf.keras.Sequential([ sparse_l1_encoder, sparse_l1_decoder ])



kl_divergence = tf.keras.losses.kullback_leibler_divergence

class KLDivergenceRegularizer(tf.keras.regularizers.Regularizer ):
    
    def __init__(self, weight, target):
        self.weight = weight
        self.target = target
        
    def __call__(self, inputs):
        mean_activities = tf.reduce_mean(inputs, axis=0)
        return self.weight * ( kl_divergence(self.target, mean_activities) +
                              kl_divergence(1. - self.target, 1.- mean_activities))



kld_reg = KLDivergenceRegularizer(weight=5e-3, target=0.1)

sparse_kl_encoder = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(300, activation="sigmoid", activity_regularizer=kld_reg),
    ])


sparse_kl_decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(28 * 28),
        tf.keras.layers.Reshape([28,28])
    ])



sparse_kl_ae = tf.keras.Sequential([ sparse_kl_encoder, sparse_kl_decoder ])


class Sampling(tf.keras.layers.Layer):
    def call( self, inputs ):
        mean, log_var = inputs
        return tf.random.normal( tf.shape(log_var) ) * tf.exp(log_var / 2) + mean
    


coding_size = 10

inputs = tf.keras.layers.Input( shape=[28, 28] )
Z = tf.keras.layers.Flatten()(inputs)
Z = tf.keras.layers.Dense(150, activation="relu")(Z)
Z = tf.keras.layers.Dense(100, activation="relu")(Z)
codings_mean = tf.keras.layers.Dense( coding_size )(Z)
codings_log_var = tf.keras.layers.Dense(coding_size)(Z)
codings = Sampling()([codings_mean, codings_log_var])

variational_encoder = tf.keras.Model( inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])

variational_encoder.summary()


decoder_inputs = tf.keras.layers.Input( shape=[coding_size] )
x = tf.keras.layers.Dense(100, activation="relu")(decoder_inputs)
x = tf.keras.layers.Dense(150, activation="relu")(x)
x = tf.keras.layers.Dense(28*28)(x)
outputs = tf.keras.layers.Reshape([28,28])(x)
variational_decoder = tf.keras.Model(inputs=[decoder_inputs], outputs=[outputs])


variational_decoder.summary()

_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = tf.keras.Model( inputs=[inputs], outputs=[reconstructions] )

variational_ae.summary()


latent_loss = -0.5 * tf.reduce_sum(
    1 + codings_log_var - tf.exp(codings_log_var) - tf.square(codings_mean),
    axis=-1)

latent_loss = -0.5 * tf.reduce_sum( 1 + codings_log_var - tf.exp(codings_log_var) - tf.square(codings_mean), axis=1 )
variational_ae.add_loss(tf.reduce_mean(latent_loss) / 784)

variational_ae.compile(loss="mse", optimizer=tf.keras.optimizers.Nadam())
history = variational_ae.fit(X_train, X_train, epochs=25, batch_size=128,
                             validation_data=(X_valid, X_valid))




codings = tf.random.normal( shape=[3*7, coding_size] )
images = variational_decoder(codings).numpy()




codings = np.zeros([7, coding_size])
codings[:, 3] = np.linspace(-0.8, 0.8, 7)
images = variational_decoder(codings).numpy()


coding_size = 30

Dense = tf.keras.layers.Dense

generator =  tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(150, activation="relu",  kernel_initializer="he_normal"),
        tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
        tf.keras.layers.Reshape([28,28])
    ])

discriminador = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(150, activation="relu",  kernel_initializer="he_normal"),
        tf.keras.layers.Dense(100, activation="sigmoid", kernel_initializer="he_normal"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

gan = tf.keras.Sequential([generator, discriminador])


discriminador.compile( loss="binary_crossentropy", optimizer="rmsprop")
discriminador.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")


batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)


def train_gan(gan, dataset, batch_size, codings_size, n_epochs):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")  # extra code
        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.train_on_batch(X_fake_and_real, y1)
            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            gan.train_on_batch(noise, y2)


train_gan(gan, dataset, batch_size, coding_size, n_epochs=50)




tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU

codings = tf.random.normal(shape=[batch_size, coding_size])
generated_images = generator.predict(codings)




coding_size = 100


generator =  tf.keras.Sequential([
        tf.keras.layers.Dense(7*7*128),
        tf.keras.layers.Reshape([7,7,128]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(64, kernel_size=5, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="same", activation="tanh")
    ])

discriminador =  tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU(0.2)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU(0.2)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

gan = tf.keras.Sequential( [generator, discriminador] )


X_train_dcgan = X_train.reshape(-1, 28, 28, 1) * 2. -1


def variance_schedule(T, s=0.008, max_beta=0.999):
    t = np.arange(T + 1)
    f = np.cost( ( t / T + s ) / (1 + s) * np.pi / 2 ) ** 2
    alpha = np.clip( f[1:] / f[:-1], 1 - max_beta, 1 )
    alpha = np.append(1, alpha).astype(np.float32)
    beta = 1 -alpha
    alpha_cumprod = np.cumprod(alpha)
    return alpha, alpha_cumprod, beta


T = 4000
alpha, alpha_cumprod, beta = variance_schedule(T)
def prepare_batch(X):
    X = tf.cast( X[..., tf.newaxis], tf.float32 ) * 2 - 1
    X_shape = tf.shape(X)
    t =tf.random.uniform([X_shape[0]], minval=1, maxval=T + 1, dtype=tf.int32)
    alpha_cm = tf.gather(alpha_cumprod, t)
    alpha_cm = tf.reshape(alpha_cm, [X_shape[0]] + [1] * (len(X_shape) - 1))
    noise = tf.random.normal(X_shape)
    return {
        "X_noisy": alpha_cm ** 0.5 * X + (1 - alpha_cm) ** 0.5 * noise,
        "time": t
        }, noise


def prepare_dataset(X, batch_size=32, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(X)
    if shuffle:
        ds = ds.shuffle(buffer_size=10_000)
    return ds.batch(batch_size).map(prepare_batch).prefetch(1)


train_set = prepare_dataset(X_train, batch_size=32, shuffle=True)
valid_set = prepare_dataset(X_valid, batch_size=32)






embed_size = 64

class TimeEncoding(tf.keras.layers.Layer):
    def __init__(self, T, embed_size, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert embed_size % 2 == 0, "embed_size must be even"
        p, i = np.meshgrid(np.arange(T + 1), 2 * np.arange(embed_size // 2))
        t_emb = np.empty((T + 1, embed_size))
        t_emb[:, ::2] = np.sin(p / 10_000 ** (i / embed_size)).T
        t_emb[:, 1::2] = np.cos(p / 10_000 ** (i / embed_size)).T
        self.time_encodings = tf.constant(t_emb.astype(self.dtype))

    def call(self, inputs):
        return tf.gather(self.time_encodings, inputs)





def build_diffusion():
    X_noisy = tf.keras.layers.Input(shape=[28,28,1], name="X_noisy")
    time_input = tf.keras.layers.Input(shape=[], dtype=tf.int32, name="time")

    time_enc = TimeEncoding(T, embed_size)(time_input)

    dim = 16
    Z = tf.keras.layers.ZeroPadding2D((3, 3))(X_noisy)
    Z = tf.keras.layers.Conv2D(dim, 3)(Z)
    Z = tf.keras.layers.BatchNormalization()(Z)
    Z = tf.keras.layers.Activation("relu")(Z)

    time = tf.keras.layers.Dense(dim)(time_enc)  # adapt time encoding
    Z = time[:, tf.newaxis, tf.newaxis, :] + Z  # add time data to every pixel

    skip = Z
    cross_skips = []  # skip connections across the down & up parts of the UNet

    for dim in (32, 64, 128):
        Z = tf.keras.layers.Activation("relu")(Z)
        Z = tf.keras.layers.SeparableConv2D(dim, 3, padding="same")(Z)
        Z = tf.keras.layers.BatchNormalization()(Z)

        Z = tf.keras.layers.Activation("relu")(Z)
        Z = tf.keras.layers.SeparableConv2D(dim, 3, padding="same")(Z)
        Z = tf.keras.layers.BatchNormalization()(Z)

        cross_skips.append(Z)
        Z = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(Z)
        skip_link = tf.keras.layers.Conv2D(dim, 1, strides=2,
                                           padding="same")(skip)
        Z = tf.keras.layers.add([Z, skip_link])

        time = tf.keras.layers.Dense(dim)(time_enc)
        Z = time[:, tf.newaxis, tf.newaxis, :] + Z
        skip = Z

    for dim in (64, 32, 16):
        Z = tf.keras.layers.Activation("relu")(Z)
        Z = tf.keras.layers.Conv2DTranspose(dim, 3, padding="same")(Z)
        Z = tf.keras.layers.BatchNormalization()(Z)

        Z = tf.keras.layers.Activation("relu")(Z)
        Z = tf.keras.layers.Conv2DTranspose(dim, 3, padding="same")(Z)
        Z = tf.keras.layers.BatchNormalization()(Z)

        Z = tf.keras.layers.UpSampling2D(2)(Z)

        skip_link = tf.keras.layers.UpSampling2D(2)(skip)
        skip_link = tf.keras.layers.Conv2D(dim, 1, padding="same")(skip_link)
        Z = tf.keras.layers.add([Z, skip_link])

        time = tf.keras.layers.Dense(dim)(time_enc)
        Z = time[:, tf.newaxis, tf.newaxis, :] + Z
        Z = tf.keras.layers.concatenate([Z, cross_skips.pop()], axis=-1)
        skip = Z

    outputs = tf.keras.layers.Conv2D(1, 3, padding="same")(Z)[:, 2:-2, 2:-2]
    return tf.keras.Model(inputs=[X_noisy, time_input], outputs=[outputs])



model = build_diffusion()
model.compile(loss=tf.keras.losses.Huber(), optimizer="nadam")
history = model.fit(train_set, validation_data=valid_set, epochs=100)


def generate(model, batch_size=32):
    X = tf.random.normal([batch_size, 28, 28, 1])
    for t in range(T, 0, -1):
        noise = (tf.random.normal if t > 1 else tf.zeros)( tf.shape(X)  )
        X_noise = model({"X_noisy": X, "time": tf.constant([t] * batch_size)})
        X = (
            1 / alpha[t] ** 0.5
            * (X - beta[t] / (1 - alpha_cumprod[t]) ** 0.5 * X_noise)
            + (1 - alpha[t]) ** 0.5 * noise
            )
        return X

tf.random.set_seed(42) 
X_gen = generate(model) 




































































































