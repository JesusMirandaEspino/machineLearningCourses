# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 15:21:20 2025

@author: jesus
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_moons
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import export_graphviz
from graphviz import Source

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



import os
print("Archivo guardado en:", os.getcwd())


iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)

output_dir = 'C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Graphviz'
output_file = 'iris_tree.dot5'
output_path = os.path.join(output_dir, output_file)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

export_graphviz(
        tree_clf,
        out_file=output_path,
        feature_names=["petal length (cm)", "petal width (cm)"],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )


Source.from_file("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Graphviz/iris_tree.dot")

tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_clf.fit(X_iris, y_iris)

output_dir = 'C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Graphviz'
output_file = 'iris_tree5.dot'
output_path = os.path.join(output_dir, output_file)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

export_graphviz(
        tree_clf,
        out_file=output_path,
        feature_names=["petal length (cm)", "petal width (cm)"],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

Source.from_file("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Graphviz/iris_tree5.dot")


tree_clf.predict([  [5,1.5] ]).round(3)

print(tree_clf.predict([  [5,1.5] ]).round(3))
print(tree_clf.predict_proba([  [5,1.5] ]).round(3))

X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)

tree_clf1 = DecisionTreeClassifier( random_state=42 )
tree_clf2 = DecisionTreeClassifier( min_samples_leaf=5, random_state=42 )
tree_clf1.fit( X_moons, y_moons )
tree_clf2.fit( X_moons, y_moons )

X_moons_test, y_moons_test = make_moons(n_samples=1000, noise=0.2, random_state=43)
tree_clf1.score( X_moons_test, y_moons_test )
tree_clf2.score( X_moons_test, y_moons_test)

np.random.seed(42)
X_quad = np.random.rand( 200,1 ) - 0.5
y_quad = X_quad ** 2 + 0.025 * np.random.rand( 200,1 )

plt.scatter(X_quad, y_quad, label='Example', color='blue')
[...]
plt.show()

tree_reg = DecisionTreeRegressor(min_samples_leaf=10,  max_depth=2, random_state=42 )
tree_reg.fit(X_quad, y_quad)

output_dir = 'C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Graphviz'
output_file = 'iris_treeA.dot'
output_path = os.path.join(output_dir, output_file)


export_graphviz(
        tree_reg,
        out_file=output_path,
        feature_names=["data"],
        class_names=["data"],
        rounded=True,
        filled=True
    )


Source.from_file("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Graphviz/iris_treeA.dot")


pca_pipeline = make_pipeline(  StandardScaler(), PCA() )
X_iris_rotated = pca_pipeline.fit_transform(X_iris)
tree_clf_pca = DecisionTreeClassifier( max_depth=2, random_state=42 )
tree_clf_pca.fit(X_iris_rotated, y_iris)
















