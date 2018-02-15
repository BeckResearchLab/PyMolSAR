from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, Imputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.svm import LinearSVR

from smdt import data_processing
from smdt import molecular_descriptors

import numpy as np
import pandas as pd


def fit_Ridge(X_train, X_test, y_train, y_test):

    # Standardization
    a = StandardScaler()
    X_train = a.fit_transform(X_train)
    X_test = a.transform(X_test)

    # Grid Search CV
    clf = RidgeCV(cv=10)
    clf.fit(X_train, y_train)

    # Metrics
    print('Training data GridSearchCV best r2 score: %.5f' % clf.score(X_train, y_train))
    print('Testing Data Classification r2 score: %.5f' % clf.score(X_test, y_test))

    return clf


def fit_ElasticNet(X_train, X_test, y_train, y_test):

    # Standardization
    a = StandardScaler()
    X_train = a.fit_transform(X_train)
    X_test = a.transform(X_test)

    # Grid Search CV
    clf = ElasticNetCV(cv=10)
    clf.fit(X_train, y_train)

    # Metrics
    print('Training data GridSearchCV best r2 score: %.5f' % clf.score(X_train, y_train))
    print('Testing Data Classification r2 score: %.5f' % clf.score(X_test, y_test))

    return clf


def fit_LinearSVR(X_train, X_test, y_train, y_test,n_features):

    # Standardization
    a = StandardScaler()
    X_train = a.fit_transform(X_train)
    X_test = a.transform(X_test)

    # Feature Selection
    b = SelectKBest(score_func=mutual_info_regression, k=n_features)
    X_train = b.fit_transform(X_train, y_train)
    X_test = b.transform(X_test)

    # Grid Search CV
    clf = LinearSVR()
    parameters = {'C':[1,5,10],'loss':['epsilon_insensitive','squared_epsilon_insensitive'],'epsilon':[0,0.1]}
    grid = GridSearchCV(clf, parameters)
    grid.fit(X_train, y_train)

    # Metrics
    print('Training data GridSearchCV best r2 score: %.5f' % grid.best_score_)
    print('Testing Data Classification r2 score: %.5f' % grid.score(X_test, y_test))

    return grid


def fit_Lasso(X_train, X_test, y_train, y_test):

    # Standardization
    a = StandardScaler()
    X_train = a.fit_transform(X_train)
    X_test = a.transform(X_test)

    # Grid Search CV
    clf = LassoCV(cv=10)
    clf.fit(X_train, y_train)

    # Metrics
    print('Training data GridSearchCV best r2 score: %.5f' % clf.score(X_train, y_train))
    print('Testing Data Classification r2 score: %.5f' % clf.score(X_test, y_test))

    return clf


def fit_RandomForestRegressor(X_train, X_test, y_train, y_test, n_features):
    # Standardization
    a = StandardScaler()
    X_train = a.fit_transform(X_train)
    X_test = a.transform(X_test)

    # Feature Selection
    b = SelectKBest(score_func=mutual_info_regression, k=n_features)
    X_train = b.fit_transform(X_train, y_train)
    X_test = b.transform(X_test)

    # Grid Search CV
    clf = RandomForestRegressor()
    parameters = {'n_estimators': [10, 100], 'criterion': ['mse', 'mae'],
                  'max_features': [1, 3, 10, 'auto', 'sqrt', 'log2'], 'oob_score': [True, False],
                  "max_depth": [3, None], "min_samples_split": [2, 3, 10], "min_samples_leaf": [1, 3, 10]}
    grid = GridSearchCV(clf, parameters)
    grid.fit(X_train, y_train)

    # Metrics
    print('Training data GridSearchCV best r2 score: %.5f' % grid.best_score_)
    print('Testing Data Regression r2 score: %.5f' % grid.score(X_test, y_test))

    return grid


