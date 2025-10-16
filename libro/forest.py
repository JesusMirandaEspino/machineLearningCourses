# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 10:42:19 2025

@author: jesus
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris, make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingRegressor, HistGradientBoostingRegressor, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder


X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)




voting_clf = VotingClassifier( estimators=[ ( 'lr', LogisticRegression(random_state=42) ),
                                             (  'rf', RandomForestClassifier(random_state=42)),
                                             (  'svc', SVC(random_state=42) ) ] )
voting_clf.fit(X_train, y_train)


for name, clf in voting_clf.named_estimators_.items():
    print( name, "=", clf.score(X_test, y_test) )


voting_clf.predict( X_test[:1] )
print(voting_clf.predict( X_test[:1] ))
print( [ clf.predict(  X_test[:1] )  for clf in voting_clf.estimators_ ] )
voting_clf.score(X_test, y_test)



voting_clf.voting = "soft"
voting_clf.named_estimators["svc"].probability = True
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)

bag_clf = BaggingClassifier( DecisionTreeClassifier(), n_estimators=500, max_samples=100, n_jobs=1, random_state=42 )
bag_clf.fit(X_train, y_train)


bag_clf = BaggingClassifier( DecisionTreeClassifier(), n_estimators=500, oob_score=True, n_jobs=1, random_state=42 )
bag_clf.fit(X_train, y_train)
print(bag_clf.oob_score_)

y_pred = bag_clf.predict(X_test)
accuracy_score( y_test, y_pred )

bag_clf.oob_decision_function_[:3]

rnd_clf = RandomForestClassifier( n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42 )
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)

accuracy_score( y_test, y_pred_rf )

bag_clf = BaggingClassifier( DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16), n_estimators=500, n_jobs=1, random_state=42 )
bag_clf.fit(X_train, y_train)

y_pred = bag_clf.predict(X_test)
accuracy_score( y_test, y_pred )


iris = load_iris(as_frame=True)
rnd_clf = RandomForestClassifier( n_estimators=500, random_state=42 )
rnd_clf.fit(iris.data, iris.target)

for score, name in zip( rnd_clf.feature_importances_, iris.data.columns ):
    print( round( score, 2 ), name )


ada_clf = AdaBoostClassifier( DecisionTreeClassifier( max_depth=1 ), n_estimators=30, learning_rate=0.5, random_state=42 )
ada_clf.fit( X_train, y_train )


ada_y_pred = ada_clf.predict(X_test)
accuracy_score( y_test, ada_y_pred )


np.random.seed(42)
X = np.random.rand( 100, 1 ) - 0.5
y = 3 * X[:,0] ** 2 + 0.05 * np.random.randn(100) 


plt.scatter(X, y, label='Example', color='blue')
[...]
plt.show()


tree_reg1 = DecisionTreeRegressor( max_depth=2, random_state=42 )
tree_reg1.fit(X, y)

y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor( max_depth=2, random_state=43 )
tree_reg2.fit(X, y2)


plt.scatter(X, y2, label='Example', color='blue')
[...]
plt.show()


y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor( max_depth=2, random_state=43 )
tree_reg3.fit(X, y3)


plt.scatter(X, y3, label='Example', color='blue')
[...]
plt.show()


X_new = np.array( [ [-0.4], [0.], [0.5] ]) 
sum( tree.predict( X_new ) for tree in ( tree_reg1, tree_reg2, tree_reg3 ) )

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=0.1, random_state=42)
gbrt.fit(X,y)



gbrt_best = GradientBoostingRegressor( max_depth=2, learning_rate=0.05, n_estimators=500, n_iter_no_change=10, random_state=42 )
gbrt_best.fit(X,y)

print(gbrt_best.n_estimators_)



housing = pd.read_csv("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/data-main/housing/housing.csv")
housing_full = pd.read_csv("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/data-main/housing/housing.csv")
housing['income_cat'] = pd.cut( housing['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5] )

splitter = StratifiedShuffleSplit( n_splits=10, test_size=0.2, random_state=42 )


strat_splits = []
for train_index, text_index, in splitter.split(housing, housing['income_cat']):
    strat_train_n = housing.iloc[train_index]
    strat_test_n = housing.iloc[text_index]
    strat_splits.append([strat_train_n, strat_test_n])
    

strat_train_set, strat_test_set = strat_splits[0]


strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, random_state=42, stratify=housing['income_cat'])

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

hgb_reg = make_pipeline( make_column_transformer( ( OrdinalEncoder(), ["ocean_proximity"] ), remainder="passthrough"), 
                         HistGradientBoostingRegressor(categorical_features=[0], random_state=42) )

hgb_reg.fit(housing, housing_labels)


stacking_clf = StackingClassifier(  estimators=[ ( 'lr', LogisticRegression( random_state=42 ) ),
                                                 ( 'rf', RandomForestClassifier(random_state=42)),
                                                 ( 'svc', SVC( probability=True, random_state=42 ) )],
                                  final_estimator=RandomForestClassifier( random_state=43 ), cv=5)

stacking_clf.fit( X_train, y_train )

stacking_clf.score(X_test, y_test)


































