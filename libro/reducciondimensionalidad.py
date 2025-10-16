# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 09:59:14 2025

@author: jesus
"""

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from scipy.spatial.transform import Rotation
from sklearn.datasets import fetch_openml, make_swiss_roll

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.random_projection import johnson_lindenstrauss_min_dim, GaussianRandomProjection
from sklearn.manifold import LocallyLinearEmbedding


m = 60
X = np.zeros((m, 3))  # initialize 3D dataset
np.random.seed(42)
angles = (np.random.rand(m) ** 3 + 0.5) * 2 * np.pi  # uneven distribution
X[:, 0], X[:, 1] = np.cos(angles), np.sin(angles) * 0.5  # oval
X += 0.28 * np.random.randn(m, 3)  # add more noise
X = Rotation.from_rotvec([np.pi / 29, -np.pi / 20, np.pi / 4]).apply(X)
X += [0.2, 0, 0.2]  # shift a bit


X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt[0]
c2 = Vt[1]

W2 = Vt[:2].T
X2D = X_centered @ W2

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)


print(pca.explained_variance_ratio_)


mnist = fetch_openml( 'mnist_784', as_frame=False )
X_train, y_train = mnist.data[:60_000], mnist.target[:60_000]
X_test, y_test = mnist.data[60_000:], mnist.target[60_000:]

pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax( cumsum >= 0.95 ) + 1

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)

pca.n_components_


plt.figure(figsize=(6, 4))
plt.plot(cumsum, linewidth=3)
plt.axis([0, 400, 0, 1])
plt.xlabel("Dimensions")
plt.ylabel("Explained Variance")


plt.plot([d, d], [0, 0.95], "k:")
plt.plot([0, d], [0.95, 0.95], "k:")

plt.plot(d, 0.95, "ko")
plt.annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7),
             arrowprops=dict(arrowstyle="->"))
plt.grid(True)
plt.show()

clf = make_pipeline( PCA(random_state=42), RandomForestClassifier(random_state=42) )
param_distrib = { "pca__n_components": np.arange(10, 80),
                 "randomforestclassifier__n_estimators": np.arange(50,500 )}

rnd_search = RandomizedSearchCV( clf, param_distrib, n_iter=10, cv=3, random_state=42 )
rnd_search.fit( X_train[:1000], y_train[:1000] )


print( rnd_search.best_params_ )

X_recovered = pca.inverse_transform(X_reduced)


plt.figure(figsize=(7, 4))
for idx, X in enumerate((X_train[::2100], X_recovered[::2100])):
    plt.subplot(1, 2, idx + 1)
    plt.title(["Original", "Compressed"][idx])
    for row in range(5):
        for col in range(5):
            plt.imshow(X[row * 5 + col].reshape(28, 28), cmap="binary",
                       vmin=0, vmax=255, extent=(row, row + 1, col, col + 1))
            plt.axis([0, 5, 0, 5])
            plt.axis("off")



rnd_pca = PCA( n_components=154, svd_solver="randomized", random_state=42 )
X_reduced = rnd_pca.fit_transform(X_train)

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)


X_reduced = inc_pca.transform(X_train)


filename = "my_mnist.mmap"
X_mmap = np.memmap( filename, dtype='float32', mode='write', shape=X_train.shape )
X_mmap[:] = X_train
X_mmap.flush()


X_mmap = np.memmap( filename,  dtype='float32', mode='readonly').reshape(-1, 784)
batch_size = X_mmap.shape[0] // n_batches
inc_pca = IncrementalPCA( n_components=154, batch_size=batch_size )
inc_pca.fit(X_mmap)  


m,e = 5_000, 0.1
d = johnson_lindenstrauss_min_dim(m, eps=e)
print(d)

n = 20_000
np.random.seed(42)
P = np.random.randn(d,n) / np.sqrt(d)

X = np.random.randn(m,n)
X_reduced = X @ P.T 

guassian_rnd_proj = GaussianRandomProjection( eps=e, random_state=42 )
X_reduced = guassian_rnd_proj.fit_transform(X)

componentes_pinv = np.linalg.pinv(guassian_rnd_proj.components_)
X_recovered = X_reduced @ componentes_pinv.T


1


























