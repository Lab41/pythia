#!/usr/bin/env python

'''
Provides ability to conduct a grid search of the hyperparameters used by the classifiers in Pythia

The output is recorded by Sacred to track experimentation results if environment variable PYTHIA_MONGO_DB_URI is set

Input
datafile (required) = Path to file containing data features from Pythia's master pipeline
targetfile (required) = Path to file containing data target values from Pythia's master pipeline
svmsearch = Boolean value to execute Grid Search on Support Vector Machine model
svmparams = Grid Search parameters for Support Vector Machine model
logregsearch = Boolean value to execute Grid Search on Logistic Regression model
logregparams = Grid Search parameters for Logistic Regression model
xgbsearch = Boolean value to execute Grid Search on XGBoost model
xgbparams = Grid Search parameters for XGBoost model

Output
Best Score = The highest average F-score calculated using the parameters provided by the user
Best Parameters = A list of the best classifier parameters based on the ranges provided by the user
Best Estimator = An explicit list of all of the best classifier's parameters
All Scores = A list of all calculated F-scores for all parameter permutations
'''

from sklearn import svm, linear_model, grid_search
import xgboost
import pickle
import os
import sys

from sacred import Experiment
from sacred.observers import MongoObserver

ex_name='pythia_gridsearch'
db_name='pythia_experiment'

def set_up_xp():
    # Check that MongoDB config is set
    try:
        mongo_uri=os.environ['PYTHIA_MONGO_DB_URI']
        ex = Experiment(ex_name)
        ex.observers.append(MongoObserver.create(url=mongo_uri,
                                         db_name=db_name))
    except KeyError as e:
        print("You must define location of MongoDB in PYTHIA_MONGO_DB_URI to record experiment output",file=sys.stderr)
        print("Proceeding without an observer! Results will not be logged!",file=sys.stderr)
        ex = Experiment(ex_name)

    return ex

xp = set_up_xp()

@xp.capture
def conduct_grid_search(datafile,targetfile,svmsearch,svmparams,logregsearch,logregparams,xgbsearch,xgbparams,_run):

    # Ensure that at least one classifier has been selected to grid search
    if [svmsearch,logregsearch,xgbsearch].count(True) == 0:
        print("Error: Grid Search requires one classifier\n")
        quit()

    # Load data files
    traindata = pickle.load( open( datafile, "rb" ) )
    traintarget = pickle.load( open( targetfile, "rb" ) )

    # Initiate classifiers and parameters as needed 
    classifiers = []
    if svmsearch:
        svmmodel = svm.SVC()
        classifiers.append(('SVM', svmmodel, svmparams))
    if logregsearch:
        logregmodel = linear_model.LogisticRegression()
        classifiers.append(("Logistic Regression", logregmodel, logregparams))
    if xgbsearch:
        xgbmodel = xgboost.XGBClassifier()
        classifiers.append(("XGBoost", xgbmodel, xgbparams))

    # Conduct grid search of selected classifiers
    for i in range(len(classifiers)):
        print("Searching " + classifiers[i][0] + " parameters...", file=sys.stderr)

        # Execute grid search
        clf = grid_search.GridSearchCV(classifiers[i][1], classifiers[i][2])
        clf.fit(traindata, traintarget)

        # If Sacred is observing, store priority Grid Search results in Mongo as info dict
        if len(xp.observers) > 0:
            _run.info[classifiers[i][0] + " Best Score"] = clf.best_score_
            _run.info[classifiers[i][0] + " Best Parameters"] = clf.best_params_
            _run.info[classifiers[i][0] + " Best Estimator"] = str(clf.best_estimator_)

        # Print all Grid Search results
        print(classifiers[i][0] + " Best Estimator",clf.best_estimator_, file=sys.stderr)
        print(classifiers[i][0] + " Best Score", clf.best_score_, file=sys.stderr)
        print(classifiers[i][0] + " Best Parameters", clf.best_params_, file=sys.stderr)
        print(classifiers[i][0] + " All Scores", file=sys.stderr)
        for score in clf.grid_scores_: print(score, file=sys.stderr)

@xp.config
def config_variables():
    # Path to file containing data features
    datafile = "data/data.pkl"

    # Path to file containing data target values
    targetfile = "data/target.pkl"

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

#@xp.main
@xp.automain
def run_experiment():
    conduct_grid_search()