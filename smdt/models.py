from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn import metrics
from smdt import utils
from smdt import data_processing
from sklearn.svm import SVR
import numpy as np


def svr_model(data, standardize=True, feature_selection=None, n_features=10, cv_metric='r2'):
    """
    Train a Support Vector Regression model
    Papameters:
        data: pandas.DataFrame
            Descriptor and Target data
        standardize: boolean, default True
            Scales features to zero mean and unit variance
        feature_selection: str, default 'univariate feature selection'
            Specify strategy for feature selection. Choose between 'remove_low_variance_features','univariate feature selection',
            'tree_based_feature_selection'
        n_features: int, default 10
            Used only in univariate feature selection
        cv_metric: str, default 'r2'
    """
    print('\nSupport Vector Regression')
    print('\nFound dataset of shape:  %s' % str(data.shape))
    print('\nData Split started...')
    train, test = utils.test_train_split(data)
    print('Train data shape: %s' % str(train.shape))
    print('Test data shape: %s' % str(test.shape))
    print('Data Split completed.')

    if standardize == True:
        print('\nData Scaling started...')
        train, test = data_processing.data_standardization(train, test)
        print('Data Scaling completed.')

    if not feature_selection == None:
        print('\nSelecting Features...')
        if feature_selection == 'low variance':
            train = data_processing.remove_low_variance_features(train)
            test = test[train.columns]
        elif feature_selection == 'univariate':
            train = data_processing.univariate_feature_selection(train, n_features)
            test = test[train.columns]
        elif feature_selection == 'tree based':
            train = data_processing.tree_based_feature_selection(train)
            test = test[train.columns]
        print('New train data shape: %s' % str(train.shape))
        print('New test data shape: %s' % str(test.shape))
        print('Feature selection completed.')

    train_descriptors, train_target = utils.descriptor_target_split(train)
    test_descriptors, test_target = utils.descriptor_target_split(test)

    parameters = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'), 'epsilon': [0.1, 1], 'C': [1e0, 1e1, 1e2, 1e3],
                  'gamma': np.logspace(-2, 2, 5)}
    model = SVR()
    print('\nGridSearchCV Parameter Grid:')
    print(parameters)
    print('\nStarted GridSearchCV on Training data...')
    clf = GridSearchCV(model, parameters, scoring=cv_metric, cv=10, refit=True).fit(np.array(train_descriptors),
                                                                                    train_target.values.ravel())
    print('GridSearchCV completed.')

    print('\nBest Estimator:')
    print('parameters: %s' % clf.best_estimator_)
    print('Cross-validation %s score of the best estimator: %.3f' % (cv_metric, clf.best_score_))
    print('\nModel Validation on Test data:')
    y_pred = clf.predict(test_descriptors)

    metric = {}
    metric['mean squared error'] = round(metrics.regression.mean_squared_error(test_target, y_pred), 3)
    metric['r2'] = round(metrics.regression.r2_score(test_target, y_pred))
    metric['mean absolute error'] = round(metrics.regression.mean_absolute_error(test_target, y_pred))
    metric['explained r2'] = round(metrics.regression.explained_variance_score(test_target, y_pred))
    metric['mean squared log error'] = round(metrics.regression.mean_squared_log_error(test_target, y_pred))
    metric['median absolute error'] = round(metrics.regression.median_absolute_error(test_target, y_pred))

    print(metric)
    return clf, metric, list(train_descriptors.columns)


def linear_model(data, standardize=True, feature_selection=None, n_features=10):
    """
    Train a Linear Regression model
    Papameters:
        data: pandas.DataFrame
            Descriptor and Target data
        standardize: boolean, default True
            Scales features to zero mean and unit variance
        feature_selection: str, default 'univariate feature selection'
            Specify strategy for feature selection. Choose between 'remove_low_variance_features','univariate feature selection',
            'tree_based_feature_selection'
        n_features: int, default 10
            Used only in univariate feature selection
    """
    print('\nLinear Regression')
    print('\nFound dataset of shape:  %s' % str(data.shape))
    print('\nData Split started...')
    train, test = utils.test_train_split(data)
    print('Train data shape: %s' % str(train.shape))
    print('Test data shape: %s' % str(test.shape))
    print('Data Split completed.')

    if standardize == True:
        print('\nData Scaling started...')
        train, test = data_processing.data_standardization(train, test)
        print('Data Scaling completed.')

    if not feature_selection == None:
        print('\nSelecting Features...')
        if feature_selection == 'low variance':
            train = data_processing.remove_low_variance_features(train)
            test = test[train.columns]
        elif feature_selection == 'univariate':
            train = data_processing.univariate_feature_selection(train, n_features)
            test = test[train.columns]
        elif feature_selection == 'tree based':
            train = data_processing.tree_based_feature_selection(train)
            test = test[train.columns]
        print('New train data shape: %s' % str(train.shape))
        print('New test data shape: %s' % str(test.shape))
        print('Feature selection completed.')

    train_descriptors, train_target = utils.descriptor_target_split(train)
    test_descriptors, test_target = utils.descriptor_target_split(test)

    model = LinearRegression()
    print('\nTraining Linear Model...')
    model.fit(train_descriptors, train_target)
    print('Training completed.')

    print('\nModel Validation on Test data:')
    y_pred = model.predict(test_descriptors)

    metric = {}
    metric['mean squared error'] = round(metrics.regression.mean_squared_error(test_target, y_pred), 3)
    metric['r2'] = round(metrics.regression.r2_score(test_target, y_pred))
    metric['mean absolute error'] = round(metrics.regression.mean_absolute_error(test_target, y_pred))
    metric['explained r2'] = round(metrics.regression.explained_variance_score(test_target, y_pred))
    metric['mean squared log error'] = round(metrics.regression.mean_squared_log_error(test_target, y_pred))
    metric['median absolute error'] = round(metrics.regression.median_absolute_error(test_target, y_pred))

    print(metric)
    return model, metric, list(train_descriptors.columns)


def random_forest_model(data, standardize=True, feature_selection=None, n_features=10, cv_metric='r2'):
    """
    Train a Random Forest Regression model
    Papameters:
        data: pandas.DataFrame
            Descriptor and Target data
        standardize: boolean, default True
            Scales features to zero mean and unit variance
        feature_selection: str, default 'univariate feature selection'
            Specify strategy for feature selection. Choose between 'remove_low_variance_features','univariate feature selection',
            'tree_based_feature_selection'
        n_features: int, default 10
            Used only in univariate feature selection
        cv_metric: str, default 'r2'
    """
    print('\nRandom Forest Regression')
    print('\nFound dataset of shape:  %s' % str(data.shape))
    print('\nData Split started...')
    train, test = utils.test_train_split(data)
    print('Train data shape: %s' % str(train.shape))
    print('Test data shape: %s' % str(test.shape))
    print('Data Split completed.')

    if standardize == True:
        print('\nData Scaling started...')
        train, test = data_processing.data_standardization(train, test)
        print('Data Scaling completed.')

    if not feature_selection == None:
        print('\nSelecting Features...')
        if feature_selection == 'low variance':
            train = data_processing.remove_low_variance_features(train)
            test = test[train.columns]
        elif feature_selection == 'univariate':
            train = data_processing.univariate_feature_selection(train, n_features)
            test = test[train.columns]
        elif feature_selection == 'tree based':
            train = data_processing.tree_based_feature_selection(train)
            test = test[train.columns]
        print('New train data shape: %s' % str(train.shape))
        print('New test data shape: %s' % str(test.shape))
        print('Feature selection completed.')

    train_descriptors, train_target = utils.descriptor_target_split(train)
    test_descriptors, test_target = utils.descriptor_target_split(test)

    parameters = {'n_estimators': [10, 50], 'criterion': ('mse', 'mae'),
                  'max_features': ('auto', 'sqrt', 'log2')}
    model = RandomForestRegressor()
    print('\nGridSearchCV Parameter Grid:')
    print(parameters)
    print('\nStarted GridSearchCV on Training data...')
    clf = GridSearchCV(model, parameters, scoring=cv_metric, cv=10, refit=True).fit(np.array(train_descriptors),
                                                                                    train_target.values.ravel())
    print('GridSearchCV completed.')

    print('\nBest Estimator:')
    print('parameters: %s' % clf.best_estimator_)
    print('Cross-validation %s score of the best estimator: %.3f' % (cv_metric, clf.best_score_))
    print('\nModel Validation on Test data:')
    y_pred = clf.predict(test_descriptors)

    metric = {}
    metric['mean squared error'] = round(metrics.regression.mean_squared_error(test_target, y_pred), 3)
    metric['r2'] = round(metrics.regression.r2_score(test_target, y_pred))
    metric['mean absolute error'] = round(metrics.regression.mean_absolute_error(test_target, y_pred))
    metric['explained r2'] = round(metrics.regression.explained_variance_score(test_target, y_pred))
    metric['mean squared log error'] = round(metrics.regression.mean_squared_log_error(test_target, y_pred))
    metric['median absolute error'] = round(metrics.regression.median_absolute_error(test_target, y_pred))

    print(metric)
    return clf, metric, list(train_descriptors.columns)


def extra_trees_model(data, standardize=True, feature_selection=None, n_features=10, cv_metric='r2'):
    """
    Train a Extra Trees Regression model
    Papameters:
        data: pandas.DataFrame
            Descriptor and Target data
        standardize: boolean, default True
            Scales features to zero mean and unit variance
        feature_selection: str, default 'univariate feature selection'
            Specify strategy for feature selection. Choose between 'remove_low_variance_features','univariate feature selection',
            'tree_based_feature_selection'
        n_features: int, default 10
            Used only in univariate feature selection
        cv_metric: str, default 'r2'
    """
    print('\nExtra Trees Regression')
    print('\nFound dataset of shape:  %s' % str(data.shape))
    print('\nData Split started...')
    train, test = utils.test_train_split(data)
    print('Train data shape: %s' % str(train.shape))
    print('Test data shape: %s' % str(test.shape))
    print('Data Split completed.')

    if standardize == True:
        print('\nData Scaling started...')
        train, test = data_processing.data_standardization(train, test)
        print('Data Scaling completed.')

    if not feature_selection == None:
        print('\nSelecting Features...')
        if feature_selection == 'low variance':
            train = data_processing.remove_low_variance_features(train)
            test = test[train.columns]
        elif feature_selection == 'univariate':
            train = data_processing.univariate_feature_selection(train, n_features)
            test = test[train.columns]
        elif feature_selection == 'tree based':
            train = data_processing.tree_based_feature_selection(train)
            test = test[train.columns]
        print('New train data shape: %s' % str(train.shape))
        print('New test data shape: %s' % str(test.shape))
        print('Feature selection completed.')

    train_descriptors, train_target = utils.descriptor_target_split(train)
    test_descriptors, test_target = utils.descriptor_target_split(test)

    parameters = {'n_estimators': [10, 50], 'criterion': ('mse', 'mae'),
                  'max_features': ('auto', 'sqrt', 'log2')}
    model = ExtraTreesRegressor()
    print('\nGridSearchCV Parameter Grid:')
    print(parameters)
    print('\nStarted GridSearchCV on Training data...')
    clf = GridSearchCV(model, parameters, scoring=cv_metric, cv=10, refit=True).fit(np.array(train_descriptors),
                                                                                    train_target.values.ravel())
    print('GridSearchCV completed.')

    print('\nBest Estimator:')
    print('parameters: %s' % clf.best_estimator_)
    print('Cross-validation %s score of the best estimator: %.3f' % (cv_metric, clf.best_score_))
    print('\nModel Validation on Test data:')
    y_pred = clf.predict(test_descriptors)

    metric = {}
    metric['mean squared error'] = round(metrics.regression.mean_squared_error(test_target, y_pred), 3)
    metric['r2'] = round(metrics.regression.r2_score(test_target, y_pred))
    metric['mean absolute error'] = round(metrics.regression.mean_absolute_error(test_target, y_pred))
    metric['explained r2'] = round(metrics.regression.explained_variance_score(test_target, y_pred))
    metric['mean squared log error'] = round(metrics.regression.mean_squared_log_error(test_target, y_pred))
    metric['median absolute error'] = round(metrics.regression.median_absolute_error(test_target, y_pred))

    print(metric)
    return clf, metric, list(train_descriptors.columns)


def gradient_boosting_model(data, standardize=True, feature_selection=None, n_features=10, cv_metric='r2'):
    """
    Train a Gradient Boosting Regression model
    Papameters:
        data: pandas.DataFrame
            Descriptor and Target data
        standardize: boolean, default True
            Scales features to zero mean and unit variance
        feature_selection: str, default 'univariate feature selection'
            Specify strategy for feature selection. Choose between 'remove_low_variance_features','univariate feature selection',
            'tree_based_feature_selection'
        n_features: int, default 10
            Used only in univariate feature selection
        cv_metric: str, default 'r2'
    """
    print('\nGradient Boosting Regression')
    print('\nFound dataset of shape:  %s' % str(data.shape))
    print('\nData Split started...')
    train, test = utils.test_train_split(data)
    print('Train data shape: %s' % str(train.shape))
    print('Test data shape: %s' % str(test.shape))
    print('Data Split completed.')

    if standardize == True:
        print('\nData Scaling started...')
        train, test = data_processing.data_standardization(train, test)
        print('Data Scaling completed.')

    if not feature_selection == None:
        print('\nSelecting Features...')
        if feature_selection == 'low variance':
            train = data_processing.remove_low_variance_features(train)
            test = test[train.columns]
        elif feature_selection == 'univariate':
            train = data_processing.univariate_feature_selection(train, n_features)
            test = test[train.columns]
        elif feature_selection == 'tree based':
            train = data_processing.tree_based_feature_selection(train)
            test = test[train.columns]
        print('New train data shape: %s' % str(train.shape))
        print('New test data shape: %s' % str(test.shape))
        print('Feature selection completed.')

    train_descriptors, train_target = utils.descriptor_target_split(train)
    test_descriptors, test_target = utils.descriptor_target_split(test)

    parameters = {'loss': ('ls', 'lad', 'huber'), 'learning_rate': [0.1, 0.5],
                  'max_depth': [3, 6, 10], 'max_features': ('auto', 'sqrt', 'log2')}
    model = GradientBoostingRegressor()
    print('\nGridSearchCV Parameter Grid:')
    print(parameters)
    print('\nStarted GridSearchCV on Training data...')
    clf = GridSearchCV(model, parameters, scoring=cv_metric, cv=10, refit=True).fit(np.array(train_descriptors),
                                                                                    train_target.values.ravel())
    print('GridSearchCV completed.')

    print('\nBest Estimator:')
    print('parameters: %s' % clf.best_estimator_)
    print('Cross-validation %s score of the best estimator: %.3f' % (cv_metric, clf.best_score_))
    print('\nModel Validation on Test data:')
    y_pred = clf.predict(test_descriptors)

    metric = {}
    metric['mean squared error'] = round(metrics.regression.mean_squared_error(test_target, y_pred), 3)
    metric['r2'] = round(metrics.regression.r2_score(test_target, y_pred))
    metric['mean absolute error'] = round(metrics.regression.mean_absolute_error(test_target, y_pred))
    metric['explained r2'] = round(metrics.regression.explained_variance_score(test_target, y_pred))
    metric['mean squared log error'] = round(metrics.regression.mean_squared_log_error(test_target, y_pred))
    metric['median absolute error'] = round(metrics.regression.median_absolute_error(test_target, y_pred))

    print(metric)
    return clf, metric, list(train_descriptors.columns)


def sgd_model(data, standardize=True, feature_selection=None, n_features=10, cv_metric='r2'):
    """
    Train a SGD Regression model
    Papameters:
        data: pandas.DataFrame
            Descriptor and Target data
        standardize: boolean, default True
            Scales features to zero mean and unit variance
        feature_selection: str, default None
            Specify strategy for feature selection. Choose between 'remove_low_variance_features','univariate feature selection',
            'tree_based_feature_selection'
        n_features: int, default 10
            Used only in univariate feature selection
        cv_metric: str, default 'r2'
    """
    print('\nSGD Regression')
    print('\nFound dataset of shape:  %s' % str(data.shape))
    print('\nData Split started...')
    train, test = utils.test_train_split(data)
    print('Train data shape: %s' % str(train.shape))
    print('Test data shape: %s' % str(test.shape))
    print('Data Split completed.')

    if standardize == True:
        print('\nData Scaling started...')
        train, test = data_processing.data_standardization(train, test)
        print('Data Scaling completed.')

    if not feature_selection == None:
        print('\nSelecting Features...')
        if feature_selection == 'low variance':
            train = data_processing.remove_low_variance_features(train)
            test = test[train.columns]
        elif feature_selection == 'univariate':
            train = data_processing.univariate_feature_selection(train, n_features)
            test = test[train.columns]
        elif feature_selection == 'tree based':
            train = data_processing.tree_based_feature_selection(train)
            test = test[train.columns]
        print('New train data shape: %s' % str(train.shape))
        print('New test data shape: %s' % str(test.shape))
        print('Feature selection completed.')

    train_descriptors, train_target = utils.descriptor_target_split(train)
    test_descriptors, test_target = utils.descriptor_target_split(test)

    parameters = {'loss': ('squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'),
                  'penalty': ('l1', 'l2', 'elasticnet', 'none')}
    model = SGDRegressor()
    print('\nGridSearchCV Parameter Grid:')
    print(parameters)
    print('\nStarted GridSearchCV on Training data...')
    clf = GridSearchCV(model, parameters, scoring=cv_metric, cv=10, refit=True).fit(np.array(train_descriptors),
                                                                                    train_target.values.ravel())
    print('GridSearchCV completed.')

    print('\nBest Estimator:')
    print('parameters: %s' % clf.best_estimator_)
    print('Cross-validation %s score of the best estimator: %.3f' % (cv_metric, clf.best_score_))
    print('\nModel Validation on Test data:')
    y_pred = clf.predict(test_descriptors)

    metric = {}
    metric['mean squared error'] = round(metrics.regression.mean_squared_error(test_target, y_pred), 3)
    metric['r2'] = round(metrics.regression.r2_score(test_target, y_pred))
    metric['mean absolute error'] = round(metrics.regression.mean_absolute_error(test_target, y_pred))
    metric['explained r2'] = round(metrics.regression.explained_variance_score(test_target, y_pred))
    metric['mean squared log error'] = round(metrics.regression.mean_squared_log_error(test_target, y_pred))
    metric['median absolute error'] = round(metrics.regression.median_absolute_error(test_target, y_pred))

    print(metric)
    return clf, metric, list(train_descriptors.columns)


def ridge_model(data, standardize=True, feature_selection=None, n_features=10, cv_metric='r2'):
    """
    Train a Ridge Regression model
    Papameters:
        data: pandas.DataFrame
            Descriptor and Target data
        standardize: boolean, default True
            Scales features to zero mean and unit variance
        feature_selection: str, default None
            Specify strategy for feature selection. Choose between 'remove_low_variance_features','univariate feature selection',
            'tree_based_feature_selection'
        n_features: int, default 10
            Used only in univariate feature selection
        cv_metric: str, default 'r2'
    """
    print('Ridge Regression')
    print('\nFound dataset of shape:  %s' % str(data.shape))
    print('\nData Split started...')
    train, test = utils.test_train_split(data)
    print('Train data shape: %s' % str(train.shape))
    print('Test data shape: %s' % str(test.shape))
    print('Data Split completed.')

    if standardize == True:
        print('\nData Scaling started...')
        train, test = data_processing.data_standardization(train, test)
        print('Data Scaling completed.')

    if not feature_selection == None:
        print('\nSelecting Features...')
        if feature_selection == 'low variance':
            train = data_processing.remove_low_variance_features(train)
            test = test[train.columns]
        elif feature_selection == 'univariate':
            train = data_processing.univariate_feature_selection(train, n_features)
            test = test[train.columns]
        elif feature_selection == 'tree based':
            train = data_processing.tree_based_feature_selection(train)
            test = test[train.columns]
        print('New train data shape: %s' % str(train.shape))
        print('New test data shape: %s' % str(test.shape))
        print('Feature selection completed.')

    train_descriptors, train_target = utils.descriptor_target_split(train)
    test_descriptors, test_target = utils.descriptor_target_split(test)

    model = RidgeCV(cv=10, scoring=cv_metric)
    print('\nStarted GridSearchCV on Training data...')
    clf = model.fit(np.array(train_descriptors), train_target.values.ravel())
    print('GridSearchCV completed.')

    print('\nBest Estimator:')
    print('alpha: %s' % clf.alpha_)
    print('\nModel Validation on Test data:')
    y_pred = clf.predict(test_descriptors)

    metric = {}
    metric['mean squared error'] = round(metrics.regression.mean_squared_error(test_target, y_pred), 3)
    metric['r2'] = round(metrics.regression.r2_score(test_target, y_pred))
    metric['mean absolute error'] = round(metrics.regression.mean_absolute_error(test_target, y_pred))
    metric['explained r2'] = round(metrics.regression.explained_variance_score(test_target, y_pred))
    metric['mean squared log error'] = round(metrics.regression.mean_squared_log_error(test_target, y_pred))
    metric['median absolute error'] = round(metrics.regression.median_absolute_error(test_target, y_pred))

    print(metric)
    return clf, metric, list(train_descriptors.columns)


def lasso_model(data, standardize=True, feature_selection=None, n_features=10):
    """
    Train a Lasso Regression model
    Papameters:
        data: pandas.DataFrame
            Descriptor and Target data
        standardize: boolean, default True
            Scales features to zero mean and unit variance
        feature_selection: str, default None
            Specify strategy for feature selection. Choose between 'remove_low_variance_features','univariate feature selection',
            'tree_based_feature_selection'
        n_features: int, default 10
            Used only in univariate feature selection
        cv_metric: str, default 'r2'
    """
    print('Lasso Regression')
    print('\nFound dataset of shape:  %s' % str(data.shape))
    print('\nData Split started...')
    train, test = utils.test_train_split(data)
    print('Train data shape: %s' % str(train.shape))
    print('Test data shape: %s' % str(test.shape))
    print('Data Split completed.')

    if standardize == True:
        print('\nData Scaling started...')
        train, test = data_processing.data_standardization(train, test)
        print('Data Scaling completed.')

    if not feature_selection == None:
        print('\nSelecting Features...')
        if feature_selection == 'low variance':
            train = data_processing.remove_low_variance_features(train)
            test = test[train.columns]
        elif feature_selection == 'univariate':
            train = data_processing.univariate_feature_selection(train, n_features)
            test = test[train.columns]
        elif feature_selection == 'tree based':
            train = data_processing.tree_based_feature_selection(train)
            test = test[train.columns]
        print('New train data shape: %s' % str(train.shape))
        print('New test data shape: %s' % str(test.shape))
        print('Feature selection completed.')

    train_descriptors, train_target = utils.descriptor_target_split(train)
    test_descriptors, test_target = utils.descriptor_target_split(test)

    model = LassoCV(cv=10)
    print('\nStarted GridSearchCV on Training data...')
    clf = model.fit(np.array(train_descriptors), train_target.values.ravel())
    print('GridSearchCV completed.')

    print('\nBest Estimator:')
    print('alpha: %s' % clf.alpha_)
    print('\nModel Validation on Test data:')
    y_pred = clf.predict(test_descriptors)

    metric = {}
    metric['mean squared error'] = round(metrics.regression.mean_squared_error(test_target, y_pred), 3)
    metric['r2'] = round(metrics.regression.r2_score(test_target, y_pred))
    metric['mean absolute error'] = round(metrics.regression.mean_absolute_error(test_target, y_pred))
    metric['explained r2'] = round(metrics.regression.explained_variance_score(test_target, y_pred))
    metric['mean squared log error'] = round(metrics.regression.mean_squared_log_error(test_target, y_pred))
    metric['median absolute error'] = round(metrics.regression.median_absolute_error(test_target, y_pred))

    print(metric)
    return clf, metric, list(train_descriptors.columns)