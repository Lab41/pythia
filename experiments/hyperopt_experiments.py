#!/usr/bin/env python

import sys
import argparse
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,allow_gc=True,lib.cnmem=0"  # Sets flags for use of GPU
from src.pipelines import parse_json, preprocess, data_gen, log_reg, svm, xgb, predict, master_pipeline
import pickle
import logging
import copy

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.initialize import Scaffold
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import src.pipelines.master_pipeline as mp
from src.pipelines.master_pipeline import main as pythia_main

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def objective(args_):

    # arguments to pass as config_updates dict
    global args
    # result to pass to hyperopt
    global result
    # command-line arguments 
    global parse_args

    try:
        ex = Experiment('Hyperopt')
        logger.debug("Adding observer for {}, DB {}".format(parse_args.mongo_db_address,parse_args.mongo_db_name))
        ex.observers.append(MongoObserver.create(url=parse_args.mongo_db_address, db_name=parse_args.mongo_db_name))
        
        pythia_args = make_args_for_pythia(args_)
        args = mp.get_args(**pythia_args) 
        ex.main(run_with_global_args)
        r = ex.run(config_updates=pythia_args)
        logger.debug("Experiment result: {}\n"
                     "Report to hyperopt: {}".format(r.result, result))

        return result

    except:
        raise
        #If we somehow cannot get to the MongoDB server, then continue with the experiment
        logger.warning("Running without Sacred")
        run_with_global_args()

args = None
result = 100.
def run_with_global_args():
    global args
    global result
    try:
        all_results = master_pipeline.main(args)
        result = -np.mean(all_results['f score'])
        return all_results
    except:
        # Have Sacred log a null result
        return None

def make_args_for_pythia(args):
    global parse_args
    print(args)
    algorithm= args['algorithm_type']

    passed_args = copy.deepcopy(args)
    passed_args.update(algorithm)
    del passed_args['algorithm_type']

    return passed_args


def run_pythia_hyperopt():

    global parse_args

    space = {
        "algorithm_type":hp.choice('algorithm_type', [
                {
                    'LOG_REG': True,
                    'LOG_C': hp.choice('log_C', [1e-5, 1e-4, 1e-3, 1e-2, 1, 10]),
                    'LOG_TOL': hp.choice('log_tol', [1e-5, 1e-4, 1e-3, 1e-2, 1, 10]),
                    'LOG_PENALTY': hp.choice('log_penalty', ["l1", "l2"])
                }, {
                    'SVM': True,
                    'SVM_C': hp.choice('svm_C', [2000, 1000]),
                    'SVM_KERNEL': hp.choice('svm_kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                    'SVM_GAMMA': hp.choice('svm_gamma', ['auto', 1000, 5000, 10000])
                }, {
                    'XGB': True,
                    'XGB_LEARNRATE': hp.choice('x_learning_rate', [0.01, 0.1, 0.5, 1]),
                    'XGB_MAXDEPTH': hp.choice('x_max_depth',[3,4,5,6]),
                    'XGB_COLSAMPLEBYTREE': hp.choice('x_colsample_bytree', [0.25, 0.5, 0.75, 1]),
                    'XGB_MINCHILDWEIGHT': hp.choice('x_colsample_bylevel', [0.25, 0.5, 0.75, 1])
                } ]),

        "BOW_APPEND":hp.choice('BOW_APPEND', [True, False]),
        "BOW_DIFFERENCE":hp.choice('BOW_DIFFERENCE', [True, False]),
        "BOW_PRODUCT":hp.choice('BOW_PRODUCT', [True, False]),
        "BOW_COS":hp.choice('BOW_COS', [True, False]),
        "BOW_TFIDF":hp.choice('BOW_TFIDF', [True, False]),
        "ST_APPEND":hp.choice('ST_APPEND', [True, False]),
        "ST_DIFFERENCE":hp.choice('ST_DIFFERENCE', [True, False]),
        "ST_PRODUCT":hp.choice('ST_PRODUCT', [True, False]),
        "ST_COS":hp.choice('ST_COS', [True, False]),
        "LDA_APPEND":hp.choice('LDA_APPEND', [True, False]),
        "LDA_DIFFERENCE":hp.choice('LDA_DIFFERENCE', [True, False]),
        "LDA_PRODUCT":hp.choice('LDA_PRODUCT', [True, False]),
        "LDA_COS":hp.choice('LDA_COS', [True, False]),
        "LDA_TOPICS":hp.choice('LDA_TOPICS', [20, 40, 100, 150, 200]),
        "W2V_APPEND":hp.choice('W2V_APPEND', [True, False]),
        "W2V_DIFFERENCE":hp.choice('W2V_DIFFERENCE', [True, False]),
        "W2V_PRODUCT":hp.choice('W2V_PRODUCT', [True, False]),
        "W2V_COS":hp.choice('W2V_COS', [True, False]),
        "W2V_AVG":hp.choice('W2V_AVG', [True, False]),
        "W2V_MAX":hp.choice('W2V_MAX', [True, False]),
        "W2V_MIN":hp.choice('W2V_MIN', [True, False]),
        "W2V_ABS":hp.choice('W2V_ABS', [True, False]),
        # Training parameters can also be set...but in order for it to go faster W2V_PRETRAINED is set to True
        'W2V_PRETRAINED': True,
    #     "W2V_MIN_COUNT",
    #     "W2V_WINDOW",
    #     "W2V_SIZE",
    #     "W2V_WORKERS",
        "CNN_APPEND":hp.choice('CNN_APPEND', [True, False]),
        "CNN_DIFFERENCE":hp.choice('CNN_DIFFERENCE', [True, False]),
        "CNN_PRODUCT":hp.choice('CNN_PRODUCT', [True, False]),
        "CNN_COS":hp.choice('CNN_COS', [True, False]),
        # As of now wordonehot doesn't play nicely with the other featurizers so it is not used
        # Also mem nets aren't in here as they run on their own
    #     "WORDONEHOT",
    #     "WORDONEHOT_VOCAB",
    #     "NOVEL_RATIO",
    #     "REPLACEMENT",
    #     "SAVEEXPERIMENTDATA",
    #     "EXPERIMENTDATAFILE",
        'directory': parse_args.directory_base,
        'RESAMPLING': True,
        "OVERSAMPLING": True,
        'USE_CACHE': True,
        "VOCAB_SIZE": hp.choice('VOCAB_SIZE', [1000, 5000, 10000, 20000]),
        "STEM": hp.choice('STEM', [True, False]),
        "FULL_VOCAB_SIZE": hp.choice('FULL_VOCAB_SIZE', [1000, 5000, 10000, 20000]),
        "FULL_VOCAB_TYPE": hp.choice('FULL_VOCAB_TYPE', ['word', 'character']),
    #     "FULL_CHAR_VOCAB",
        'SAVE_RESULTS': True
    }
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=int(parse_args.num_runs), trials=trials)
    print("Best run ", best)
    return trials, best

if __name__ == '__main__':
    """
    Runs a series of test using Hyperopt to determine which parameters are most important
    All tests will be logged using Sacred - if the sacred database is set up correctly, otherwise it will simply run
    To run pass in the number of hyperopt runs, the mongo db address and name, as well as the directory of files to test
    For example for 10 tests:
    python experiments/hyperopt_experiments.py 10 db_server:00000 pythia data/stackexchange/anime
    """


    parser = argparse.ArgumentParser(description="Pythia Hyperopt Tests logging to Sacred")
    parser.add_argument("num_runs", type=int, help="Number of Hyperopt Runs")
    parser.add_argument("mongo_db_address", type=str, help="Address of the Mongo DB")
    parser.add_argument("mongo_db_name", type=str, help="Name of the Mongo DB")
    parser.add_argument("directory_base", type=str, help="Directory of files")

    global parse_args
    parse_args = parser.parse_args()

    if int(parse_args.num_runs)<=0:
        print("Must have more than one run")

    # Monkey patch to avoid having to declare all our variables
    def noop(item):
        pass
    Scaffold._warn_about_suspicious_changes = noop

    trial_results, best = run_pythia_hyperopt()
    with open( "pythia_hyperopt_results" + '.pkl', 'wb') as f:
        pickle.dump(trial_results, f, pickle.HIGHEST_PROTOCOL)
