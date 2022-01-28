from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer as Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, recall_score, make_scorer, classification_report, roc_curve, precision_recall_curve, precision_score, recall_score, accuracy_score, matthews_corrcoef, jaccard_score as jaccard_similarity_score, zero_one_loss, auc, roc_auc_score
from smdt import metrics
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

from imblearn.over_sampling import RandomOverSampler


def get_modelName(clf):
    model_name = []
    for i in str(clf):
        if i == '(':
            break
        model_name += i
    model_name = "".join(model_name)
    return model_name


def fit_RandomForestClassifier(X_train, X_test, y_train, y_test, target_label):
    # Pipeline
    a = Imputer(missing_values='NaN', strategy='median', axis=0)
    b = StandardScaler()
    c = SelectKBest()
    d = RandomOverSampler()
    X_res, y_res = d.fit_sample(X_train, y_train)
    clf = RandomForestClassifier()
    model = Pipeline([('impute', a), ('scaling', b), ('anova', c), ('rf', clf)])

    # Grid Search CV
    parameters = {'anova__k': [5,10,20,40],
                  'rf__n_estimators': [10, 50], 'rf__criterion': ['gini', 'entropy'],
                  'rf__max_features': ['auto', 'sqrt']}
    grid = GridSearchCV(model, parameters, cv=10, scoring='f1_weighted')
    grid.fit(X_res, y_res)

    # Features Used
    final_pipeline = grid.best_estimator_
    select_indices = final_pipeline.named_steps['anova'].transform(np.arange(X_train.shape[1]).reshape(1, -1))
    feature_names = X_train.columns[select_indices]
    # Predicting and scoring on test set
    class_index = list(grid.classes_).index(target_label)
    y_pred = grid.predict(X_test)
    y_score = grid.predict_proba(X_test)[:, class_index]
    model_name = get_modelName(clf)

    # Plot
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(121)
    # ROC curve
    metrics.plot_roc(y_test, y_score, target_label, model_name)
    # Precision Recall Curve
    plt.subplot(122)
    metrics.plot_prc(y_test, y_score, target_label, model_name)
    plt.show()

    # Get Metrics
    metric = metrics.get_ClassificationMetrics(model_name, y_test, y_pred, y_score, target_label)
    print('Training data best accuracy: %.5f' % grid.best_score_)
    print('Testing data accuracy: %.5f' % grid.score(X_test, y_test))
    print()
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    return metric, feature_names, y_score


def fit_LinearSVC(X_train, X_test, y_train, y_test, target_label):

    y_train = y_train.map({target_label: 1})
    y_train[y_train != 1] = 0
    y_test = y_test.map({target_label: 1})
    y_test[y_test != 1] = 0
    target_label = 1

    parameters = [{'C': [1, 10, 100], 'penalty': ['l1', 'l2'], 'dual': [False], 'loss': ['squared_hinge'], 'class_weight': ['balanced']},
                  {'C': [1, 10, 100], 'penalty': ['l2'], 'dual': [True], 'loss': ['hinge'], 'class_weight': ['balanced']}]

    # Pipeline
    a = Imputer(missing_values='NaN', strategy='median', axis=0)
    b = StandardScaler()
    c = SelectKBest()
    d = RandomOverSampler()
    X_res, y_res = d.fit_sample(X_train, y_train)
    clf = LinearSVC()
    model = Pipeline([('impute', a), ('scaling', b), ('anova', c), ('rf', clf)])

    # Grid Search CV
    parameters = [{'anova__k': [5,10,20,40], 'rf__C': [1, 10, 100], 'rf__penalty': ['l1', 'l2'], 'rf__dual': [False], 'rf__loss': ['squared_hinge'],
                   'rf__class_weight': ['balanced']},
                  {'anova__k': [5,10,20,40], 'rf__C': [1, 10, 100], 'rf__penalty': ['l2'], 'rf__dual': [True], 'rf__loss': ['hinge'],
                   'rf__class_weight': ['balanced']}]

    grid = GridSearchCV(model, parameters, cv=10, scoring='f1_weighted')
    grid.fit(X_res, y_res)

    # Features Used
    final_pipeline = grid.best_estimator_
    select_indices = final_pipeline.named_steps['anova'].transform(np.arange(X_train.shape[1]).reshape(1, -1))
    feature_names = X_train.columns[select_indices]
    # Predicting and scoring on test set
    y_pred = grid.predict(X_test)
    y_score = grid.decision_function(X_test)
    model_name = get_modelName(clf)

    # Plot
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(121)
    # ROC curve
    metrics.plot_roc(y_test, y_score, grid.classes_[1], model_name)
    # Precision Recall Curve
    plt.subplot(122)
    metrics.plot_prc(y_test, y_score, grid.classes_[1], model_name)
    plt.show()

    # Get Metrics
    metric = metrics.get_ClassificationMetrics(model_name, y_test, y_pred, y_score, grid.classes_[1])
    print('Training data best accuracy: %.5f' % grid.best_score_)
    print('Testing data accuracy: %.5f' % grid.score(X_test, y_test))
    print()
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    return metric, feature_names, y_score


def fit_GaussianNB(X_train, X_test, y_train, y_test, target_label):

    # Pipeline
    a = Imputer(missing_values='NaN', strategy='median', axis=0)
    b = StandardScaler()
    c = SelectKBest()
    d = RandomOverSampler()
    X_res, y_res = d.fit_sample(X_train, y_train)
    clf = GaussianNB()
    model = Pipeline([('impute', a), ('scaling', b), ('anova', c), ('rf', clf)])

    # Grid Search CV
    parameters = {'anova__k': [5,10,20,40]}
    grid = GridSearchCV(model, parameters, cv=10, scoring='f1_weighted')
    grid.fit(X_res, y_res)

    # Features Used
    final_pipeline = grid.best_estimator_
    select_indices = final_pipeline.named_steps['anova'].transform(np.arange(X_train.shape[1]).reshape(1, -1))
    feature_names = X_train.columns[select_indices]
    # Predicting and scoring on test set
    class_index = list(grid.classes_).index(target_label)
    y_pred = grid.predict(X_test)
    y_score = grid.predict_proba(X_test)[:, class_index]
    model_name = get_modelName(clf)

    # Plot
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(121)
    # ROC curve
    metrics.plot_roc(y_test, y_score, target_label, model_name)
    # Precision Recall Curve
    plt.subplot(122)
    metrics.plot_prc(y_test, y_score, target_label, model_name)
    plt.show()

    # Get Metrics
    metric = metrics.get_ClassificationMetrics(model_name, y_test, y_pred, y_score, target_label)
    print('Training data best accuracy: %.5f' % grid.best_score_)
    print('Testing data accuracy: %.5f' % grid.score(X_test, y_test))
    print()
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    return metric, feature_names, y_score


def fit_KNearestNeighbors(X_train, X_test, y_train, y_test, target_label):
    # Pipeline
    a = Imputer(missing_values='NaN', strategy='median', axis=0)
    b = StandardScaler()
    c = SelectKBest()
    d = RandomOverSampler()
    X_res, y_res = d.fit_sample(X_train, y_train)
    clf = KNeighborsClassifier()
    model = Pipeline([('impute', a), ('scaling', b), ('anova', c), ('rf', clf)])

    # Grid Search CV
    parameters = {'anova__k': [5,10,20,40],
                  'rf__n_neighbors': [5, 8, 10], 'rf__weights': ['uniform', 'distance'], 'rf__p': [1, 2],
                  'rf__algorithm': ['auto', 'ball_tree', 'kd_tree']}
    grid = GridSearchCV(model, parameters, cv=10, scoring='f1_weighted')
    grid.fit(X_res, y_res)

    # Features Used
    final_pipeline = grid.best_estimator_
    select_indices = final_pipeline.named_steps['anova'].transform(np.arange(X_train.shape[1]).reshape(1, -1))
    feature_names = X_train.columns[select_indices]
    # Predicting and scoring on test set
    class_index = list(grid.classes_).index(target_label)
    y_pred = grid.predict(X_test)
    y_score = grid.predict_proba(X_test)[:, class_index]
    model_name = get_modelName(clf)

    # Plot
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(121)
    # ROC curve
    metrics.plot_roc(y_test, y_score, target_label, model_name)
    # Precision Recall Curve
    plt.subplot(122)
    metrics.plot_prc(y_test, y_score, target_label, model_name)
    plt.show()

    # Get Metrics
    metric = metrics.get_ClassificationMetrics(model_name, y_test, y_pred, y_score, target_label)
    print('Training data best accuracy: %.5f' % grid.best_score_)
    print('Testing data accuracy: %.5f' % grid.score(X_test, y_test))
    print()
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    return metric, feature_names, y_score


def fit_SGDClassifier(X_train, X_test, y_train, y_test, target_label):

    y_train = y_train.map({target_label: 1})
    y_train[y_train != 1] = 0
    y_test = y_test.map({target_label: 1})
    y_test[y_test != 1] = 0
    target_label = 1

    # Pipeline
    a = Imputer(missing_values='NaN', strategy='median', axis=0)
    b = StandardScaler()
    c = SelectKBest()
    d = RandomOverSampler()
    X_res, y_res = d.fit_sample(X_train, y_train)
    clf = SGDClassifier()
    model = Pipeline([('impute', a), ('scaling', b), ('anova', c), ('rf', clf)])

    # Grid Search CV
    parameters = {'anova__k': [5,10,20,40],
                  'rf__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                  'rf__alpha': [0.0001, 0.00001],'rf__class_weight': ['balanced']}

    grid = GridSearchCV(model, parameters, cv=10, scoring='f1_weighted')
    grid.fit(X_res, y_res)

    # Features Used
    final_pipeline = grid.best_estimator_
    select_indices = final_pipeline.named_steps['anova'].transform(np.arange(X_train.shape[1]).reshape(1, -1))
    feature_names = X_train.columns[select_indices]
    # Predicting and scoring on test set
    y_pred = grid.predict(X_test)
    y_score = grid.decision_function(X_test)
    model_name = get_modelName(clf)

    # Plot
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(121)
    # ROC curve
    metrics.plot_roc(y_test, y_score, grid.classes_[1], model_name)
    # Precision Recall Curve
    plt.subplot(122)
    metrics.plot_prc(y_test, y_score, grid.classes_[1], model_name)
    plt.show()

    # Get Metrics
    metric = metrics.get_ClassificationMetrics(model_name, y_test, y_pred, y_score, grid.classes_[1])
    print('Training data best accuracy: %.5f' % grid.best_score_)
    print('Testing data accuracy: %.5f' % grid.score(X_test, y_test))
    print()
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    return metric, feature_names, y_score


# define baseline model
def baseline_model(input_dim,optimizer):
    # create model
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def fit_MLPClassifier(X_train, X_test, y_train, y_test, target_label):

    # Pipeline
    a = Imputer(missing_values='NaN', strategy='median', axis=0)
    b = StandardScaler()
    c = SelectKBest()
    d = RandomOverSampler()
    X_res, y_res = d.fit_sample(X_train, y_train)
    clf = KerasClassifier(build_fn=baseline_model,verbose=0)
    model = Pipeline([('impute', a), ('scaling', b), ('anova', c), ('rf', clf)])

    # Grid Search CV
    parameters = [{'anova__k': [5], 'rf__input_dim':[5], 'rf__epochs':[10, 50],
                  'rf__batch_size': [10, 20], 'rf__optimizer': ['SGD', 'Adam', 'rmsprop']},
                  {'anova__k': [10], 'rf__input_dim': [10], 'rf__epochs': [10, 50],
                   'rf__batch_size': [10, 20], 'rf__optimizer': ['SGD', 'Adam', 'rmsprop']},
                  {'anova__k': [20], 'rf__input_dim': [20], 'rf__epochs': [10, 50],
                   'rf__batch_size': [10, 20], 'rf__optimizer': ['SGD', 'Adam', 'rmsprop']},
                  {'anova__k': [40], 'rf__input_dim': [40], 'rf__epochs': [10, 50],
                   'rf__batch_size': [10, 20], 'rf__optimizer': ['SGD', 'Adam', 'rmsprop']}
                  ]

    grid = GridSearchCV(model, parameters, cv=10, scoring='f1_weighted')
    grid.fit(X_res, y_res)

    # Features Used
    final_pipeline = grid.best_estimator_
    select_indices = final_pipeline.named_steps['anova'].transform(np.arange(X_train.shape[1]).reshape(1, -1))
    feature_names = X_train.columns[select_indices]
    # Predicting and scoring on test set
    class_index = list(grid.classes_).index(target_label)
    y_pred = grid.predict(X_test)
    y_score = grid.predict_proba(X_test)[:, class_index]
    model_name = 'MLPClassifier'

    # Plot
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(121)
    # ROC curve
    metrics.plot_roc(y_test, y_score, target_label, model_name)
    # Precision Recall Curve
    plt.subplot(122)
    metrics.plot_prc(y_test, y_score, target_label, model_name)
    plt.show()

    # Get Metrics
    metric = metrics.get_ClassificationMetrics(model_name, y_test, y_pred, y_score, target_label)
    print('Training data best accuracy: %.5f' % grid.best_score_)
    print('Testing data accuracy: %.5f' % grid.score(X_test, y_test))
    print()
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    return metric, feature_names, y_score

