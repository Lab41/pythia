#!/usr/bin/env python

'''
Implements auto-sklearn on features extracted by Pythia project's master pipeline

Note - auto-sklearn works on Ubuntu/Linux but will not build on Mac or Windows, see http://automl.github.io/auto-sklearn/stable/index.html

The output is recorded by Sacred if a MongoObserver is passed in via command line (-m HOST:PORT:MY_DB)

Input
experimentdatafile (required) = Path to file containing data features from Pythia's master pipeline
resamplingstrategy = How to handle overfitting. Options are holdout, holdout-interative-fit, cv, nested-cv, partial-cv
resamplingstrategyarguments = Additional arguments for resampling strategy. See http://automl.github.io/auto-sklearn/stable/api.html 
timefortask = Time limit allowed for auto-sklearn to find models

Output
results (dict) = The models selected by auto-sklearn's optimization 
'''

import autosklearn.classification
import pickle
import os
import sys
import numpy
import logging
from src.utils import performance_metrics

from sacred import Experiment
from sacred.observers import MongoObserver

def set_up_xp():
    ex_name = 'pythia_auto_sklearn'
    ex = Experiment(ex_name)

    return ex

xp = set_up_xp()

@xp.capture
def conduct_auto_sklearn(experimentdatafile,resamplingstrategy,resamplingstrategyarguments,timefortask):

    # Load data files
    lunchbox = pickle.load(open(experimentdatafile,"rb"))

    # Set up autosklearn and run against Pythia feature sets
    clf = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timefortask, per_run_time_limit=360, \
                     initial_configurations_via_metalearning=25, ensemble_size=50, ensemble_nbest=50, seed=1, \
                     ml_memory_limit=3000, include_estimators=None, include_preprocessors=None, \
                     resampling_strategy=resamplingstrategy, resampling_strategy_arguments=resamplingstrategyarguments, \
                     tmp_folder=None, output_folder=None, delete_tmp_folder_after_terminate=True, \
                     delete_output_folder_after_terminate=True, shared_mode=False)
    
    # TODO Find way to suppress voluminous INFO messages from autosklearn
    clf.fit(numpy.asarray(lunchbox['train_data']), numpy.asarray(lunchbox['train_target']))

    # Print autoasklearn results
    print("Models",clf.show_models(), file=sys.stderr)

    # Get performance metrics of autolearn models against testing data
    predictions = clf.predict(numpy.asarray(lunchbox['test_data']))
    performresults = performance_metrics.get_perform_metrics(numpy.asarray(lunchbox['test_target']), predictions)

    # Fill results dictionary to return to Sacred for logging
    results = dict()
    results['autosklearn_models'] = clf.show_models() 
    results['autosklearn_perform_results'] = performresults
    results['directory'] = lunchbox['directory']
    results['features'] = lunchbox['features']
    results['algorithms'] = lunchbox['algorithms']
    results['parameters'] = lunchbox['parameters'] 

    return results

@xp.config
def config_variables():
    # Path to file containing data features
    experimentdatafile = "data/experimentdatafile.pkl"

    # Parameters for autosklearn classifier
    resamplingstrategy = 'holdout'
    resamplingstrategyarguments = None
    timefortask = 3600 

@xp.automain
def run_experiment():
    return conduct_auto_sklearn()
