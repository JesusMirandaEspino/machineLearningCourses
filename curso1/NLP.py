# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 13:52:01 2025

@author: jesus
"""

import numpy as np # aplicacion de matematicas
import matplotlib.pyplot as plt # mostrar datos
import pandas as pd # manipular datos
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

nltk.download('stopwords')
corpus = []
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
#print(corpus)


cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


cm = confusion_matrix(y_test, y_pred)
print("GaussianNB")
print(cm)
accuracy_score(y_test, y_pred)




classifier2 = LogisticRegression(random_state = 0)
classifier2.fit(X_train, y_train)
y_pred2 = classifier2.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred2)
print("regression")
print(cm2)
accuracy_score(y_test, y_pred2)




classifier3 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier3.fit(X_train, y_train)
y_pred3 = classifier3.predict(X_test)
cm3 = confusion_matrix(y_test, y_pred3)
print("K nearest")
print(cm3)
accuracy_score(y_test, y_pred3)



classifier4 = SVC(kernel = 'linear', random_state = 0)
classifier4.fit(X_train, y_train)
y_pred4 = classifier4.predict(X_test)
print("SVC")
cm4 = confusion_matrix(y_test, y_pred4)
print(cm4)



classifier5 = SVC(kernel = 'rbf', random_state = 0)
classifier5.fit(X_train, y_train)
y_pred5 = classifier5.predict(X_test)
print("Kernel SVC")
cm5 = confusion_matrix(y_test, y_pred5)
print(cm5)
accuracy_score(y_test, y_pred5)





classifier6 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier6.fit(X_train, y_train)
y_pred6 = classifier6.predict(X_test)
cm6 = confusion_matrix(y_test, y_pred6)
print("Tree Classification")
print(cm6)
accuracy_score(y_test, y_pred6)



classifier7 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier7.fit(X_train, y_train)
y_pred7 = classifier.predict(X_test)
print("Random Forest")
cm7 = confusion_matrix(y_test, y_pred7)
print(cm7)


classifier8 = DecisionTreeRegressor(criterion = 'friedman_mse', random_state = 0)
classifier8.fit(X_train, y_train)
y_pred8 = classifier6.predict(X_test)
cm8 = confusion_matrix(y_test, y_pred8)
print("DecisionTreeRegressor")
print(cm8)
accuracy_score(y_test, y_pred6)





classifier9 = DecisionTreeClassifier(criterion = 'entropy', max_depth=3, random_state = 0)
classifier9.fit(X_train, y_train)
y_pred9 = classifier9.predict(X_test)
cm9 = confusion_matrix(y_test, y_pred9)
print("Tree Classification")
print(cm9)
accuracy_score(y_test, y_pred6)



