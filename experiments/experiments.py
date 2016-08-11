#!/usr/bin/env python

import sys
import os

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

    # DIRECTORY
    directory = 'stack_exchange_data/corpus_filtered/movies'

    # FEATURES
    # bag of words
    BOW_APPEND = True
    BOW_DIFFERENCE = False
    BOW_PRODUCT = False
    BOW_COS = False
    BOW_TFIDF = False

    # skipthoughts
    ST_APPEND = False
    ST_DIFFERENCE = False
    ST_PRODUCT = False
    ST_COS = False

    # lda
    LDA_APPEND = False
    LDA_DIFFERENCE = False
    LDA_PRODUCT = False
    LDA_COS = False
    LDA_TOPICS = 50

    #word2vec
    W2V_APPEND = False
    W2V_DIFFERENCE = False
    W2V_PRODUCT = False
    W2V_COS = False
    W2V_PRETRAINED = False
    W2V_MIN_COUNT = 5
    W2V_WINDOW = 5
    W2V_SIZE = 100
    W2V_WORKERS = 3

    #one-hot CNN layer
    CNN_APPEND = False
    CNN_DIFFERENCE = False
    CNN_PRODUCT = False
    CNN_COS = False
    #The vocabulary can either be character or word
    #If words, WORDONEHOT_VOCAB will be used as the vocab length
    CNN_VOCAB_TYPE = "character"
    CNN_CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789"

    # wordonehot (will not play nicely with other featurization methods b/c not
    # vector)
    WORDONEHOT = False
    #WORDONEHOT_DOCLENGTH = None
    WORDONEHOT_VOCAB = 5000

    # ALGORITHMS
    # logistic regression
    LOG_REG = False
    LOG_PENALTY = 'l2'
    LOG_TOL = 1e-4
    LOG_C = 1e-4

    # svm
    SVM = False
    SVM_C = 2000
    SVM_KERNEL = 'linear'
    SVM_GAMMA = 'auto'

    # xgboost
    XGB = True
    XGB_LEARNRATE = 0.1
    XGB_MAXDEPTH = 3
    XGB_MINCHILDWEIGHT = 1
    XGB_COLSAMPLEBYTREE = 1

    # PARAMETERS
    # resampling
    RESAMPLING = True
    NOVEL_RATIO = None
    OVERSAMPLING = False
    REPLACEMENT = False

    #vocabulary
    VOCAB_SIZE = 1000
    STEM = False

    SEED = None

@xp.main
def run_experiment(directory,
            BOW_APPEND,
            BOW_DIFFERENCE,
            BOW_PRODUCT,
            BOW_COS,
            BOW_TFIDF,
            ST_APPEND,
            ST_DIFFERENCE,
            ST_PRODUCT,
            ST_COS,
            LDA_APPEND,
            LDA_DIFFERENCE,
            LDA_PRODUCT,
            LDA_COS,
            LDA_TOPICS,
            W2V_APPEND,
            W2V_DIFFERENCE,
            W2V_PRODUCT,
            W2V_COS,
            W2V_PRETRAINED,
            W2V_MIN_COUNT,
            W2V_WINDOW,
            W2V_SIZE,
            W2V_WORKERS,
            CNN_APPEND,
            CNN_DIFFERENCE,
            CNN_PRODUCT,
            CNN_COS,
            CNN_VOCAB_TYPE,
            CNN_CHAR_VOCAB,
            WORDONEHOT,
            WORDONEHOT_VOCAB,
            LOG_REG,
            LOG_PENALTY,
            LOG_TOL,
            LOG_C,
            SVM,
            SVM_C,
            SVM_KERNEL,
            SVM_GAMMA,
            XGB,
            XGB_LEARNRATE,
            XGB_MAXDEPTH,
            XGB_MINCHILDWEIGHT,
            XGB_COLSAMPLEBYTREE,
            RESAMPLING,
            NOVEL_RATIO,
            OVERSAMPLING,
            REPLACEMENT,
            VOCAB_SIZE,
            STEM,
            SEED):
    return pythia_main(
        get_args(
            directory,
            BOW_APPEND,
            BOW_DIFFERENCE,
            BOW_PRODUCT,
            BOW_COS,
            BOW_TFIDF,
            ST_APPEND,
            ST_DIFFERENCE,
            ST_PRODUCT,
            ST_COS,
            LDA_APPEND,
            LDA_DIFFERENCE,
            LDA_PRODUCT,
            LDA_COS,
            LDA_TOPICS,
            W2V_APPEND,
            W2V_DIFFERENCE,
            W2V_PRODUCT,
            W2V_COS,
            W2V_PRETRAINED,
            W2V_MIN_COUNT,
            W2V_WINDOW,
            W2V_SIZE,
            W2V_WORKERS,
            CNN_APPEND,
            CNN_DIFFERENCE,
            CNN_PRODUCT,
            CNN_COS,
            CNN_VOCAB_TYPE,
            CNN_CHAR_VOCAB,
            WORDONEHOT,
            WORDONEHOT_VOCAB,
            LOG_REG,
            LOG_PENALTY,
            LOG_TOL,
            LOG_C,
            SVM,
            SVM_C,
            SVM_KERNEL,
            SVM_GAMMA,
            XGB,
            XGB_LEARNRATE,
            XGB_MAXDEPTH,
            XGB_MINCHILDWEIGHT,
            XGB_COLSAMPLEBYTREE,
            RESAMPLING,
            NOVEL_RATIO,
            OVERSAMPLING,
            REPLACEMENT,
            VOCAB_SIZE,
            STEM,
            SEED)
    )

if __name__=="__main__":
    xp.run_commandline()
