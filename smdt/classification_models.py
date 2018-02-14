from sklearn import datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder, Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from smdt import data_processing
from smdt import molecular_descriptors
from sklearn import metrics

import pandas as pd


def fit_RandomForestClassifier(X, y, n_features):
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    # Standardization
    a = StandardScaler()
    X_train = a.fit_transform(X_train)
    X_test = a.transform(X_test)

    # Feature Selection
    b = SelectKBest(k=n_features)
    X_train = b.fit_transform(X_train, y_train)
    X_test = b.transform(X_test)

    # Grid Search CV
    clf = RandomForestClassifier()
    parameters = {'n_estimators': [10, 100], 'criterion': ['gini', 'entropy'],
                  'max_features': ['auto', 'sqrt', 'log2'], 'oob_score': [True, False], 'verbose': [0]}
    grid = GridSearchCV(clf, parameters)
    grid.fit(X_train, y_train)

    # Metrics
    print('Training data GridSearchCV accuracy: %.5f' % grid.best_score_)
    print('Testing Data Classification accuracy: %.5f' % grid.score(X_test, y_test))
    print()
    print('Classification Report:')
    print(metrics.classification_report(y_test, grid.predict(X_test)))

    return grid


def fit_LinearSVC(X, y, n_features):
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    # Standardization
    a = StandardScaler()
    X_train = a.fit_transform(X_train)
    X_test = a.transform(X_test)

    # Feature Selection
    b = SelectKBest(k=n_features)
    X_train = b.fit_transform(X_train, y_train)
    X_test = b.transform(X_test)
    features = b.get_support(indices=True)


    # Grid Search CV
    clf = LinearSVC()
    parameters = [{'C': [1, 10, 100], 'penalty': ['l1', 'l2'], 'dual': [False], 'loss': ['squared_hinge']},
                  {'C': [1, 10, 100], 'penalty': ['l2'], 'dual': [True], 'loss': ['hinge']}]
    grid = GridSearchCV(clf, parameters, cv=10)
    grid.fit(X_train, y_train)

    # Metrics
    print('Training data GridSearchCV accuracy: %.5f' % grid.best_score_)
    print('Testing Data Classification accuracy: %.5f' % grid.score(X_test, y_test))
    print()
    print('Classification Report:')
    print(metrics.classification_report(y_test, grid.predict(X_test)))

    return grid


def fit_GaussianNB(X, y, n_features):
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    # Standardization
    a = StandardScaler()
    X_train = a.fit_transform(X_train)
    X_test = a.transform(X_test)

    # Feature Selection
    b = SelectKBest(k=n_features)
    X_train = b.fit_transform(X_train, y_train)
    X_test = b.transform(X_test)

    # Grid Search CV
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # Metrics
    print('Training data GridSearchCV accuracy: %.5f' % clf.score(X_train, y_train))
    print('Testing Data Classification accuracy: %.5f' % clf.score(X_test, y_test))
    print()
    print('Classification Report:')
    print(metrics.classification_report(y_test, clf.predict(X_test)))

    return clf


def fit_KNearestNeighbors(X, y, n_features):
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    # Standardization
    a = StandardScaler()
    X_train = a.fit_transform(X_train)
    X_test = a.transform(X_test)

    # Feature Selection
    b = SelectKBest(k=n_features)
    X_train = b.fit_transform(X_train, y_train)
    X_test = b.transform(X_test)

    # Grid Search CV
    clf = KNeighborsClassifier()

    parameters = [{'n_neighbors': [5, 8, 10], 'weights': ['uniform', 'distance'], 'p': [1, 2],
                   'algorithm': ['auto', 'ball_tree', 'kd_tree']}]
    grid = GridSearchCV(clf, parameters, cv=10)
    grid.fit(X_train, y_train)

    # Metrics
    print('Training data GridSearchCV accuracy: %.5f' % grid.best_score_)
    print('Testing Data Classification accuracy: %.5f' % grid.score(X_test, y_test))
    print()
    print('Classification Report:')
    print(metrics.classification_report(y_test, grid.predict(X_test)))

    return clf


def fit_SGDClassifier(X, y, n_features):
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    # Standardization
    a = StandardScaler()
    X_train = a.fit_transform(X_train)
    X_test = a.transform(X_test)

    # Feature Selection
    b = SelectKBest(k=n_features)
    X_train = b.fit_transform(X_train, y_train)
    X_test = b.transform(X_test)

    # Grid Search CV
    clf = SGDClassifier()

    parameters = [
        {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 'alpha': [0.0001, 0.00001]}]
    grid = GridSearchCV(clf, parameters, cv=10)
    grid.fit(X_train, y_train)

    # Metrics
    print('Training data GridSearchCV accuracy: %.5f' % grid.best_score_)
    print('Testing Data Classification accuracy: %.5f' % grid.score(X_test, y_test))
    print()
    print('Classification Report:')
    print(metrics.classification_report(y_test, grid.predict(X_test)))

    return clf