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

    a = Imputer(missing_values='NaN', strategy='median', axis=0)
    b = StandardScaler()
    c = SelectKBest(score_func=mutual_info_regression)
    clf = RidgeCV(cv=10)
    model = Pipeline([('impute', a), ('scaling', b), ('anova', c), ('rf', clf)])

    # Grid Search CV
    parameters = {'anova__k': [5, 10, 20, 40]}

    grid = GridSearchCV(model, parameters)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    # Metrics
    print('r2 score: %.5f' % clf.score(X_test, y_test))
    print('explained r2 score: %.5f' % metrics.explained_variance_score(X_test, y_test))
    print('mean absolute error: %.5f' % metrics.mean_absolute_error(X_test, y_test))
    print('mean squared error: %.5f' % metrics.mean_squared_error(X_test, y_test))
    print('mean squared log error: %.5f' % metrics.median_absolute_error(X_test, y_test))
    print('median absolute error: %.5f' % metrics.r2_score(X_test, y_test))

    return grid, y_pred


def fit_ElasticNet(X_train, X_test, y_train, y_test):

    a = Imputer(missing_values='NaN', strategy='median', axis=0)
    b = StandardScaler()
    c = SelectKBest(score_func=mutual_info_regression)
    clf = ElasticNetCV(cv=10)
    model = Pipeline([('impute', a), ('scaling', b), ('anova', c), ('rf', clf)])

    # Grid Search CV
    parameters = {'anova__k': [5, 10, 20, 40]}

    grid = GridSearchCV(model, parameters)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    # Metrics
    print('r2 score: %.5f' % clf.score(X_test, y_test))
    print('explained r2 score: %.5f' % metrics.explained_variance_score(X_test, y_test))
    print('mean absolute error: %.5f' % metrics.mean_absolute_error(X_test, y_test))
    print('mean squared error: %.5f' % metrics.mean_squared_error(X_test, y_test))
    print('mean squared log error: %.5f' % metrics.median_absolute_error(X_test, y_test))
    print('median absolute error: %.5f' % metrics.r2_score(X_test, y_test))

    return grid, y_pred


def fit_LinearSVR(X_train, X_test, y_train, y_test,n_features):

    a = Imputer(missing_values='NaN', strategy='median', axis=0)
    b = StandardScaler()
    c = SelectKBest(score_func=mutual_info_regression)
    clf = LinearSVR()
    model = Pipeline([('impute', a), ('scaling', b), ('anova', c), ('rf', clf)])

    # Grid Search CV
    parameters = {'anova__k': [5, 10, 20, 40],
                  'C':[1,5,10],'loss':['epsilon_insensitive','squared_epsilon_insensitive'],'epsilon':[0,0.1]}
    grid = GridSearchCV(model, parameters)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    # Metrics
    print('r2 score: %.5f' % clf.score(X_test, y_test))
    print('explained r2 score: %.5f' % metrics.explained_variance_score(X_test, y_test))
    print('mean absolute error: %.5f' % metrics.mean_absolute_error(X_test, y_test))
    print('mean squared error: %.5f' % metrics.mean_squared_error(X_test, y_test))
    print('mean squared log error: %.5f' % metrics.median_absolute_error(X_test, y_test))
    print('median absolute error: %.5f' % metrics.r2_score(X_test, y_test))

    return grid, y_pred


def fit_Lasso(X_train, X_test, y_train, y_test):

    a = Imputer(missing_values='NaN', strategy='median', axis=0)
    b = StandardScaler()
    c = SelectKBest(score_func=mutual_info_regression)
    clf = LassoCV(cv=10)
    model = Pipeline([('impute', a), ('scaling', b), ('anova', c), ('rf', clf)])

    # Grid Search CV
    parameters = {'anova__k': [5, 10, 20, 40]}

    grid = GridSearchCV(model, parameters)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    # Metrics
    print('r2 score: %.5f' % clf.score(X_test, y_test))
    print('explained r2 score: %.5f' % metrics.explained_variance_score(X_test, y_test))
    print('mean absolute error: %.5f' % metrics.mean_absolute_error(X_test, y_test))
    print('mean squared error: %.5f' % metrics.mean_squared_error(X_test, y_test))
    print('mean squared log error: %.5f' % metrics.median_absolute_error(X_test, y_test))
    print('median absolute error: %.5f' % metrics.r2_score(X_test, y_test))

    return grid, y_pred


def fit_RandomForestRegressor(X_train, X_test, y_train, y_test, n_features):

    a = Imputer(missing_values='NaN', strategy='median', axis=0)
    b = StandardScaler()
    c = SelectKBest(score_func=mutual_info_regression)
    clf = RandomForestRegressor()
    model = Pipeline([('impute', a), ('scaling', b), ('anova', c), ('rf', clf)])

    # Grid Search CV
    parameters = {'anova__k': [5,10,20,40],
                  'rf__n_estimators': [10, 100], 'rf__criterion': ['mse', 'mae'],
                  'rf__max_features': [1, 3, 10, 'auto', 'sqrt', 'log2'], 'rf__oob_score': [True, False],
                  "rf__max_depth": [3, None], "rf__min_samples_split": [2, 3, 10], "rf__min_samples_leaf": [1, 3, 10]}
    grid = GridSearchCV(model, parameters)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    # Metrics
    print('r2 score: %.5f' % clf.score(X_test, y_test))
    print('explained r2 score: %.5f' % metrics.explained_variance_score(X_test, y_test))
    print('mean absolute error: %.5f' % metrics.mean_absolute_error(X_test, y_test))
    print('mean squared error: %.5f' % metrics.mean_squared_error(X_test, y_test))
    print('mean squared log error: %.5f' % metrics.median_absolute_error(X_test, y_test))
    print('median absolute error: %.5f' % metrics.r2_score(X_test, y_test))

    return grid, y_pred


