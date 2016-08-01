#!/usr/bin/env python

import sys
import os
import pprint
import subprocess
import json

from sacred import Experiment
from sacred.observers import MongoObserver

from src.pipelines.master_pipeline import main as pythia_main
from src.pipelines.master_pipeline import get_args

ex_name='pythia_experiment'
db_name='pythia_experiment'

def set_up_xp():
    # Check that MongoDB config is set
    try:
        mongo_uri=os.environ['PYTHIA_MONGO_DB_URI']
    except KeyError as e:
        print("Must define location of MongoDB in PYTHIA_MONGO_DB_URI for observer output",file=sys.stderr)
        raise

    ex = Experiment(ex_name)
    ex.observers.append(MongoObserver.create(url=mongo_uri,
                                         db_name=db_name))
    return ex


xp = set_up_xp()

@xp.config
def config_variables():

    args = get_args(

    # DIRECTORY
    directory = 'stack_exchange_data/corpus_filtered/movies',

    # FEATURES
    # bag of words
    BOW_APPEND = True,
    BOW_DIFFERENCE = True,
    BOW_PRODUCT = True,
    BOW_COS = True,
    BOW_TFIDF = True,
    BOW_VOCAB = 10000,

    # skipthoughts
    ST_APPEND = False,
    ST_DIFFERENCE = False,
    ST_PRODUCT = False,
    ST_COS = False,

    # lda
    LDA_APPEND = False,
    LDA_DIFFERENCE = False,
    LDA_PRODUCT = False,
    LDA_COS = False,
    LDA_VOCAB = 10000,
    LDA_TOPICS = 50,

    # ALGORITHMS
    # logistic regression
    LOG_REG = False,
    LOG_PENALTY = 'l2',
    LOG_TOL = 1e-4,
    LOG_C = 1e-4,

    # svm
    SVM = False,
    SVM_C = 2000,
    SVM_KERNAL = 'linear',
    SVM_GAMMA = 'auto',

    # xgboost
    XGB = True,
    XGB_LEARNRATE = 0.1,
    XGB_MAXDEPTH = 3,
    XGB_MINCHILDWEIGHT = 1,
    XGB_COLSAMPLEBYTREE = 1,

    # PARAMETERS
    # resampling
    RESAMPLING = True,
    NOVEL_RATIO = None,
    OVERSAMPLING = False,
    REPLACEMENT = False,

    SEED = None)

@xp.main
def run_experiment(args):

    return pythia_main(args)

if __name__=="__main__":
    xp.run_commandline()
