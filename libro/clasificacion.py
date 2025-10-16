# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 08:56:27 2025

@author: jesus
"""

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, precision_recall_curve, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.multioutput import ClassifierChain


mnist = fetch_openml("mnist_784", as_frame=False)
X, y = mnist.data, mnist.target

def plot_digit(image_data):
    image = image_data.reshape(28,28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    
    
some_digit= X[0]
plot_digit(some_digit)
plt.show()
    
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

print( sgd_clf.predict([some_digit]) )

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


print( cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy" ) )


dummy_clf = DummyClassifier()
dummy_clf.fit( X_train, y_train_5 )

print( dummy_clf.predict(X_train) )

print( cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy" ) )

y_train_pred = cross_val_predict( sgd_clf, X_train, y_train_5, cv=3 )

cm = confusion_matrix( y_train_5, y_train_pred )

y_train_perfect_predictions = y_train_5

cm_perfect = confusion_matrix( y_train_5, y_train_perfect_predictions )


precision_score(y_train_5, y_train_pred)

recall_score(y_train_5, y_train_pred)


f1_score(y_train_5, y_train_pred)

y_scores = sgd_clf.decision_function([some_digit])

print(y_scores)


threshold = 0
y_some_digit_predict = ( y_scores > threshold )
print(y_some_digit_predict)



threshold = 3000
y_some_digit_predict = ( y_scores > threshold )
print(y_some_digit_predict)


y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


plt.plot( thresholds, precisions[:-1], "b--", label="Presicion", linewidth=2  )
plt.plot( thresholds, recalls[:-1], "g-", label="Recall", linewidth=2  )
plt.vlines( threshold, 0, 1.0, "k", "dotted", label="Treshold"  )
[...]
plt.show()



plt.plot( recalls, precisions, label="Presicion/Recalls", linewidth=2  )
[...]
plt.show()

idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]

print(threshold_for_90_precision)

y_train_pred_90 = ( y_scores >= threshold_for_90_precision )
print(y_train_pred_90)

precision_score(y_train_5, y_train_pred_90)

recall_at_90_presicion = recall_score(y_train_5, y_train_pred_90)
print(recall_at_90_presicion)

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


idx_for_threshold_at_90 = ( thresholds <= threshold_for_90_precision ).argmax()
tpr_90, fpr_90 = tpr[ idx_for_threshold_at_90 ], fpr[idx_for_threshold_at_90]

plt.plot( fpr, tpr, linewidth=2, label="ROC curve")
plt.plot( [0,1], [0,1], "k:", label="Random Classifier's ROC curve")
plt.plot( [fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
[...]
plt.show()

roc_auc_score(y_train_5, y_scores)

forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")


y_scores_forest = y_probas_forest[:,1]
precision_forest, recalls_forest, threshold_forest = precision_recall_curve(y_train_5, y_scores_forest)


plt.plot( recalls_forest, precision_forest, "b-", linewidth=2, label="Random Forest")
plt.plot( recalls, precisions, "--", label="SDG", linewidth=2  )
[...]
plt.show()

y_train_pred_forest = y_probas_forest[:,1] >= 0.5
f1_score( y_train_5, y_train_pred_forest)

roc_auc_score(y_train_5, y_scores_forest)


svm_clf = SVC(random_state=42)
svm_clf.fit( X_train[:2000], y_train[:2000] )

svm_clf.predict([some_digit])

some_digit_scores = svm_clf.decision_function([some_digit])
print(some_digit_scores.round(2))

class_id = some_digit_scores.argmax()
print(class_id)

print(svm_clf.classes_)

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit( X_train[:2000], y_train[:2000])

ovr_clf.predict([some_digit])
print(len(ovr_clf.estimators_))

sdg_clf = SGDClassifier(random_state=42)
sdg_clf.fit(X_train, y_train)
sdg_clf.predict([some_digit])

sdg_clf.decision_function([some_digit])
cross_val_score(sdg_clf, X_train, y_train, scoring="accuracy")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
print(X_train_scaled)
cross_val_score(sdg_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

y_train_pred = cross_val_predict(sdg_clf, X_train_scaled, y_train, cv=3)
print("termino")
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()



ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, normalize="true", values_format=".0%")
plt.show()

sample_weight = ( y_train_pred != y_train )

ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, sample_weight=sample_weight, normalize="true", values_format=".0%")
plt.show()

y_train_large = (y_train >= '7')
y_train_odd = (y_train.astype('int8') % 2 == 1) 
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")

chain_clf = ClassifierChain( SVC(), cv=3, random_state=42 )
chain_clf.fit( X_train[:2000], y_multilabel[:2000] )

chain_clf.predict([some_digit])


np.random.seed(42)
noise = np.random.randint( 0, 100, (len(X_train), 784) )
X_train_mod = X_train + noise

noise =  np.random.randint( 0, 100, (len(X_test), 784) )
X_test_mod = X_test + noise

y_train_mod = X_train
y_test_mod = X_test

knn_clf = KNeighborsClassifier()
knn_clf.fit( y_train_mod, y_train_mod )

clean_digit = knn_clf.predict([X_test_mod[0]])
plot_digit(clean_digit)
plt.show()





