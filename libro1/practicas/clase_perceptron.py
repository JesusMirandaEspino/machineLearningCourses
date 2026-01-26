# This is a sample Python script.

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np

class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float32(0.0)
        self.error = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.error.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)



    v1 = np.array([1,2,3])
    v2 = 0.5 * v1

    print( np.arccos( v1.dot(v2) / ( np.linalg.norm(v1) * np.linalg.norm(v2) ) ) )


df = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/libro1/iris.data', header=None, encoding="utf-8")

print(df.tail())