from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, Imputer
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split, GridSearchCV
from smdt import data_processing
from smdt import molecular_descriptors
from sklearn import metrics
import numpy as np
import pandas as pd



def fit_RandomForestRegressor(X, y, n_features):
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Standardization
    a = StandardScaler()
    X_train = a.fit_transform(X_train)
    X_test = a.transform(X_test)

    # Feature Selection
    b = SelectKBest(k=n_features)
    X_train = b.fit_transform(X_train, y_train)
    X_test = b.transform(X_test)

    # Grid Search CV
    clf = RandomForestRegressor()
    parameters = {'n_estimators': [10, 100], 'criterion': ['gini', 'entropy'],
                  'max_features': ['auto', 'sqrt', 'log2'], 'oob_score': [True, False], 'verbose': [0]}
    grid = GridSearchCV(clf, parameters)
    grid.fit(X_train, y_train)

    # Metrics
    print('Training data GridSearchCV best r2 score: %.5f' % grid.best_score_)
    print('Testing Data Classification r2 score: %.5f' % grid.score(X_test, y_test))

    return grid
