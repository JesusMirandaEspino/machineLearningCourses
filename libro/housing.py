# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 16:06:11 2025

@author: jesus
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from zlib import crc32

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer, make_column_selector, make_column_transformer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
import joblib

from scipy import stats


housing = pd.read_csv("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/data-main/housing/housing.csv")
housing_full = pd.read_csv("C:/Users/jesus/IA/machinelearning/machinelearningbook/archivos/data-main/housing/housing.csv")

print(housing.info())


print(housing.value_counts())


print(housing['ocean_proximity'].value_counts())


print(housing.describe())


housing.hist( bins=50, figsize=(12,8) )


def suffle_and_split_data(data, test_ratio):
    suffled_indices = np.random.permutation(len(data))
    test_set_size = int( len(data) * test_ratio)
    test_indices = suffled_indices[:test_set_size]
    train_indices = suffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]



train_set, test_set = suffle_and_split_data(housing, 0.2)
print(len(train_set))
print(len(test_set))


def is_id_test_set( identifier, test_ratio ):
    return crc32( np.int64( identifier )) < test_ratio * 2**32


def split_data_with_id_hash( data, test_ratio, id_column ):
    ids = data[id_column]
    int_test_set = ids.apply( lambda id_: is_id_test_set(id_, test_ratio) )
    return data.loc[~int_test_set], data.loc[int_test_set]



housing_with_id = housing.reset_index()
train_set, test_set = split_data_with_id_hash( housing_with_id, 0.2, "index" )


housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"] * 1000
train_set, test_set = split_data_with_id_hash( housing_with_id, 0.2, "id" )




train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)



housing['income_cat'] = pd.cut( housing['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5] )

housing['income_cat'].value_counts().sort_index().plot.bar( rot=0, grid=True )
plt.xlabel("Income category")
plt.ylabel("NUmber of district")
plt.show()


splitter = StratifiedShuffleSplit( n_splits=10, test_size=0.2, random_state=42 )
strat_splits = []
for train_index, text_index, in splitter.split(housing, housing['income_cat']):
    strat_train_n = housing.iloc[train_index]
    strat_test_n = housing.iloc[text_index]
    strat_splits.append([strat_train_n, strat_test_n])
    
    
    
strat_train_set, strat_test_set = strat_splits[0]


strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, random_state=42, stratify=housing['income_cat'])


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()
housing.plot( kind="scatter", x='longitude', y="latitude", grid=True )


housing.plot( kind="scatter", x='longitude', y="latitude", grid=True, alpha=0.2 )
plt.show()


housing.plot( kind="scatter", x='longitude', y="latitude", grid=True, s=housing["population"] / 100, label="Population",
             c="median_house_value", cmap="jet", colorbar=True, legend=True, sharex=False, figsize=(10,7))
plt.show()

housing_ = housing.copy()
housing_.drop("ocean_proximity",  axis=1, inplace=True)
corr_matrix = housing_.corr()
print( corr_matrix["median_house_value"].sort_values(ascending=False) )


attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))


housing.plot( kind="scatter", x='median_income', y="median_house_value", grid=True, alpha=0.1 )
plt.show()


housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]



housing_ = housing.copy()
housing_.drop("ocean_proximity",  axis=1, inplace=True)
corr_matrix = housing_.corr()
print( corr_matrix["median_house_value"].sort_values(ascending=False) )


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()



imputer = SimpleImputer( strategy="median" )

housing_num = housing.select_dtypes( include=[np.number] )

imputer.fit(housing_num)


print(imputer.statistics_)
print(housing_num.median().values)


X = imputer.transform(housing_num)

housing_str = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
housing_cat = housing[["ocean_proximity"]]

print(housing_cat.head(8))


ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

print(ordinal_encoder.categories_)

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

housing_cat_1hot.toarray()


df_test = pd.DataFrame( {"ocean_proximity": ["INLAND", "NEAR BAY"]} )
pd.get_dummies(df_test)

cat_encoder.transform(df_test)

df_test_unknown = pd.DataFrame( {"ocean_proximity": [">2H OCEAN", "ISLAND"]} )
pd.get_dummies(df_test_unknown)

cat_encoder.handle_unknown = "ignore"
cat_encoder.transform(df_test_unknown)


print(cat_encoder.transform(df_test))

print(cat_encoder.transform(df_test_unknown))

print(cat_encoder.feature_names_in_)

print(cat_encoder.get_feature_names_out())


print(df_test_unknown.index)



#df_output = pd.DataFrame(cat_encoder.transform(df_test_unknown),  columns=cat_encoder.get_feature_names_out(), index=df_test_unknown.index)


mina_max_scaler = MinMaxScaler( feature_range=(-1, 1) )
housing_num_max_scaled = mina_max_scaler.fit_transform(housing_num)


std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

age_simil_35 = rbf_kernel( housing[["housing_median_age"]], [[35]], gamma=0.1 )


target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform( housing_labels.to_frame() )


model = LinearRegression()
model.fit( housing[["median_income"]], scaled_labels )
some_new_data = housing[["median_income"]].iloc[:5]

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)

model = TransformedTargetRegressor( LinearRegression(), transformer=StandardScaler() )
model.fit(housing[["median_income"]],  housing_labels)
predictions = model.predict(some_new_data)


log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])


rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[.35]], gamma=0.1))
age_simil = rbf_transformer.transform(housing[["housing_median_age"]])

sf_coords= 37.7749, -122.41
rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = rbf_transformer.transform(housing[["latitude", "longitude"]])

ratio_transformer = FunctionTransformer(lambda X: X[:,[0]] / X[:,[1]])
print( ratio_transformer.transform(  np.array( [ [1.,2.], [3.,4.] ] ) ) )



class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):
        self.with_mean = with_mean
        
    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    
    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_



class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        
        
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self
    
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_features_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]], sample_weight=housing_labels)


print(similarities[:3].round(2))



num_pipeline = Pipeline([ ("impute", SimpleImputer( strategy="median" )),
                         ("standardize", StandardScaler()) ])



num_pipeline = make_pipeline( SimpleImputer( strategy="median" ), StandardScaler() )



housing_num_prepared = num_pipeline.fit_transform(housing_num)

print(housing_num_prepared[:2].round(2))


df_housing_num_prepared = pd.DataFrame( housing_num_prepared, columns=num_pipeline.get_feature_names_out(), index=housing_num.index )


num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]

cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline( SimpleImputer( strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")  )


preprocessing = ColumnTransformer([ ("num", num_pipeline, num_attribs), ("cat", cat_pipeline, cat_attribs) ])

preprocessing = make_column_transformer( ( num_pipeline, make_column_selector(dtype_include=np.number)),
                                         ( cat_pipeline, make_column_selector(dtype_include=object )) 
                                    )


housing_prepared = preprocessing.fit_transform(housing)




def column_ratio(X):
    return X[:,[0]] / X[:,[1]]

def ratio_name(function_transformer, feature_names_in_):
    return ["ratio"]


def ratio_pipeline():
    return make_pipeline( SimpleImputer(strategy="median"), 
                             FunctionTransformer(column_ratio, feature_names_out=ratio_name ), 
                             StandardScaler() 
                        )


log_pipeline = make_pipeline( SimpleImputer( strategy="median" ),
                              FunctionTransformer(np.log, feature_names_out="one-to-one"),
                              StandardScaler())

cluster_simil = ClusterSimilarity( n_clusters=10, gamma=1., random_state=42 )
default_num_pipeline = make_pipeline( SimpleImputer(strategy="median"), StandardScaler() )

preprocessing = ColumnTransformer([ 
        ( "bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"] ),
        ( "rooms_per_house", ratio_pipeline(), ["total_rooms", "households"] ),
        ( "peopple_per_house", ratio_pipeline(), ["population", "households"] ),
        ( "log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"] ),
        ( "geo", cluster_simil, ["latitude", "longitude"] ),
        ( "cat", cat_pipeline, make_column_selector(dtype_include=object) )
    ],
    remainder=default_num_pipeline)



housing_prepared = preprocessing.fit_transform(housing)


print(housing_prepared.shape)

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

housing_predictions = lin_reg.predict(housing)

print( housing_predictions[:5].round(-2) )
print( housing_labels.iloc[:5].values )


lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_mse_root = root_mean_squared_error(housing_labels, housing_predictions)


print(lin_mse)
print(lin_mse_root)


tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)


housing_predictions = tree_reg.predict(housing)



lin_mse_tree = mean_squared_error(housing_labels, housing_predictions)
lin_mse_root_tree = root_mean_squared_error(housing_labels, housing_predictions)


print(lin_mse_tree)
print(lin_mse_root_tree)


tree_rmse = -cross_val_score(tree_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)

print(pd.Series(tree_rmse).describe())



forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)


print(pd.Series(forest_rmses).describe())


full_pipeline = Pipeline( [ ("preprocessing", preprocessing ), ("random_forest", RandomForestRegressor(random_state=42) )])

param_grid = [ { "preprocessing__geo__n_clusters": [5,8,10], "random_forest__max_features": [4,6,8] },
                  {"preprocessing__geo__n_clusters": [10,15], "random_forest__max_features": [6,8,10]} ]


grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error")
grid_search.fit(housing, housing_labels)



print(grid_search.best_params_)
print(grid_search.best_estimator_)
print(grid_search.cv_results_)


cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values( by="mean_test_score", ascending=False, inplace=True )

print(cv_res.head())

param_distribs = { "preprocessing__geo__n_clusters": randint(low=3, high=50),
                  "random_forest__max_features": randint(low=2, high=20)}

rnd_search = RandomizedSearchCV( full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3, 
                                scoring="neg_root_mean_squared_error", random_state=42)

rnd_search.fit(housing, housing_labels)

print("termino")

final_model = rnd_search.best_estimator_
feature_importances = final_model["random_forest"].feature_importances_
print(feature_importances.round(2))


#print( sorted(zip( feature_importances, final_model["preprocessing"].get_feature_names_out() ), reverse=True) )



X_test = strat_test_set.drop(["median_house_value"], axis=1)
y_test = strat_test_set["median_house_value"].copy()



final_predictions = final_model.predict(X_test)



final_mse = mean_squared_error(y_test, final_predictions)
finañ_mse_root_ = root_mean_squared_error(y_test, final_predictions)


print(final_mse)
print(finañ_mse_root_)

confidence = 0.95
squared_errors = (final_predictions - y_test ) ** 2

print( np.sqrt( stats.t.interval(confidence, len(squared_errors) - 1, loc=squared_errors.mean(),
                                 scale=stats.sem(squared_errors) )))



joblib.dump(final_model, "my_california_housing_model.pkl")

d




















