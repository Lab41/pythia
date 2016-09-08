#!/usr/bin/env python

'''
Conducts a grid search of the hyperparameters used by the classifiers in Pythia (logreg, svm, xgboost)

The output is recorded by Sacred if a MongoObserver is passed in via command line (-m HOST:PORT:MY_DB)

Input
experimentdatafile (required) : Path to file containing data features from Pythia's master pipeline
svmsearch : Boolean value to execute Grid Search on Support Vector Machine model
svmparams : Grid Search parameters for Support Vector Machine model
logregsearch : Boolean value to execute Grid Search on Logistic Regression model
logregparams : Grid Search parameters for Logistic Regression model
xgbsearch : Boolean value to execute Grid Search on XGBoost model
xgbparams : Grid Search parameters for XGBoost model
allscores : Boolean value to print all calculated Fscores to stderr and pass to Sacred

Output
results (dict) : Dictionary containing best score, best parameters, best estimator from Grid Search
and metadata about the data file that was examined by Grid Search. Results dict is recorded by Sacred.
'''

from sklearn import svm, linear_model, grid_search
import xgboost
import pickle
import os
import sys

from sacred import Experiment
from sacred.observers import MongoObserver

def set_up_xp():

    ex_name = 'pythia_gridsearch'
    ex = Experiment(ex_name)

    return ex

xp = set_up_xp()

@xp.capture
def conduct_grid_search(experimentdatafile,svmsearch,svmparams,logregsearch,logregparams,xgbsearch,xgbparams,allscores):

    # Ensure that only one classifier has been selected to grid search
    test = [svmsearch,logregsearch,xgbsearch]
    if test.count(True) == 0 or test.count(True) > 1:
        print("Error: Grid Search requires one classifier\n")
        quit()

    # Initiate classifiers and parameters as needed 
    if svmsearch:
        svmmodel = svm.SVC()
        classifier=['SVM', svmmodel, svmparams]
    elif logregsearch:
        logregmodel = linear_model.LogisticRegression()
        classifier=["Logistic Regression", logregmodel, logregparams]
    elif xgbsearch:
        xgbmodel = xgboost.XGBClassifier()
        classifier=["XGBoost", xgbmodel, xgbparams]

    print("Searching " + classifier[0] + " parameters...", file=sys.stderr)

    # Load data files
    lunchbox = pickle.load(open(experimentdatafile,"rb"))

    # Conduct grid search of selected classifier
    clf = grid_search.GridSearchCV(classifier[1], classifier[2])
    clf.fit(lunchbox['train_data'], lunchbox['train_target'])

    results = dict()
    results["gridsearch_classifier"] = classifier[0]
    results["gridsearch_best_params"] = clf.best_params_
    results["gridsearch_best_score"] = clf.best_score_
    results["gridsearch_best_estimator"] = str(clf.best_estimator_)
    results['directory'] = lunchbox['directory']
    results['features'] = lunchbox['features']
    results['algorithms'] = lunchbox['algorithms']
    results['parameters'] = lunchbox['parameters']

    # Print all Grid Search results
    print("Best Estimator",clf.best_estimator_, file=sys.stderr)
    print("Best Score", clf.best_score_, file=sys.stderr)
    print("Best Parameters", clf.best_params_, file=sys.stderr)

    if allscores:
        print("All Scores", file=sys.stderr)
        for score in clf.grid_scores_:
            print(score, file=sys.stderr)
        results['allscores'] = str(clf.grid_scores_)

    return results

@xp.config
def config_variables():
    # Path to file containing data features
    experimentdatafile = "data/experimentdatafile.pkl"

    # Boolean value to execute Grid Search on Support Vector Machine model
    svmsearch = False

    # Grid Search parameters for Support Vector Machine model
    svmparams = {'kernel':['linear', 'rbf', 'poly'], \
              'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], \
              'gamma': ['auto', 0.01, 0.001, 0.0001, 0.0001]}

    # Boolean value to execute Grid Search on Logistic Regression model
    logregsearch = False

    # Grid Search parameters for Logistic Regression model
    logregparams = {'penalty':['l1', 'l2'], \
              'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000], \
              'tol': [0.01, 0.001, 0.0001, 0.0001, 0.00001]}

    # Boolean value to execute Grid Search on XGBoost model
    xgbsearch = False

    # Grid Search parameters for XGBoost model
    xgbparams = {'learning_rate':[.001, .01, .1, .2, .5], \
              'max_depth':[3, 5, 10, 50, 100], \
              'min_child_weight': [2, 5, 10, 50, 100]}

    # Boolean value to print all scores from Grid Search
    allscores = False

@xp.automain
def run_experiment():
    return conduct_grid_search()