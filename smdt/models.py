from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from smdt import feature_selection
import sklearn
import numpy as np

def build_linear(df_x,df_y,n_features):
    selected_features,new_df_x=feature_selection.tree_based_feature_selection(df_x,df_y,n_features)
    clf = LinearRegression(n_jobs=-1)
    scores = cross_val_score(clf, new_df_x,df_y, cv=10,scoring='r2')
    return np.mean(scores)

def build_random_forest(df_x,df_y,n_features):
    selected_features,new_df_x=feature_selection.tree_based_feature_selection(df_x,df_y,n_features)
    clf = RandomForestRegressor(n_jobs=-1)
    scores = cross_val_score(clf, new_df_x,df_y, cv=10,scoring='r2')
    return np.mean(scores)

def build_lasso(df_x,df_y,n_features):
    selected_features,new_df_x=feature_selection.tree_based_feature_selection(df_x,df_y,n_features)
    clf = Lasso()
    scores = cross_val_score(clf, new_df_x,df_y, cv=10,scoring='r2')
    return np.mean(scores)

def build_elastic_net(df_x,df_y,n_features):
    selected_features,new_df_x=feature_selection.tree_based_feature_selection(df_x,df_y,n_features)
    clf = ElasticNet()
    scores = cross_val_score(clf, new_df_x,df_y, cv=10,scoring='r2')
    return np.mean(scores)

def build_ridge(df_x,df_y,n_features):
    selected_features,new_df_x=feature_selection.tree_based_feature_selection(df_x,df_y,n_features)
    clf = Ridge()
    scores = cross_val_score(clf, new_df_x,df_y, cv=10,scoring='r2')
    return np.mean(scores)

def build_linear_SVR(df_x,df_y,n_features):
    selected_features,new_df_x=feature_selection.tree_based_feature_selection(df_x,df_y,n_features)
    clf = SVR(kernel = 'linear')
    scores = cross_val_score(clf, new_df_x,df_y, cv=10,scoring='r2')
    return np.mean(scores)
