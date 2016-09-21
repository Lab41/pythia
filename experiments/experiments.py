#!/usr/bin/env python

import sys
import os

from sacred import Experiment
from sacred.observers import MongoObserver

from src.pipelines.master_pipeline import main as pythia_main
from src.pipelines.master_pipeline import get_args

'''
Conducts an experiment on Pythia's master pipeline using Sacred

The output is recorded by Sacred if a MongoObserver is passed in via command line (-m HOST:PORT:MY_DB)
'''

def set_up_xp():

    ex_name='pythia_experiment'
    ex = Experiment(ex_name)

    return ex

xp = set_up_xp()

@xp.config
def config_variables():

    # DIRECTORY
    directory = 'data/stackexchange/anime'
    #directory = 'stack_exchange_data/corpus/scifi'

    # FEATURES
    # bag of words
    BOW_APPEND = False
    BOW_DIFFERENCE = False
    BOW_PRODUCT = False
    BOW_COS = False
    BOW_TFIDF = False
    BOW_BINARY = True

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
    
    #Mem_nets
    MEM_NET = False
    MEM_VOCAB = 50
    MEM_TYPE = 'dmn_basic'
    MEM_BATCH = 1
    MEM_EPOCHS = 5
    MEM_MASK_MODE = 'word'
    MEM_EMBED_MODE = 'word2vec'
    MEM_ONEHOT_MIN_LEN = 140
    MEM_ONEHOT_MAX_LEN = 1000

    #word2vec
    # If AVG, MAX, MIN or ABS are selected, APPEND, DIFFERENCE, PRODUCT or COS must be selected
    W2V_AVG = False
    W2V_MAX = False
    W2V_MIN = False
    W2V_ABS = False
    # If APPEND, DIFFERENCE, PRODUCT or COS are selected AVG, MAX, MIN or ABS must be selected
    W2V_APPEND = False
    W2V_DIFFERENCE = False
    W2V_PRODUCT = False
    W2V_COS = False
    W2V_PRETRAINED = False
    W2V_MIN_COUNT = 5
    W2V_WINDOW = 5
    # W2V_SIZE should be set to 300 if using the Google News pretrained word2vec model
    W2V_SIZE = 300
    W2V_WORKERS = 3

    #one-hot CNN layer
    #The one-hot CNN will use the full_vocab parameters
    CNN_APPEND = False
    CNN_DIFFERENCE = False
    CNN_PRODUCT = False
    CNN_COS = False

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

    SGD = False
    SGD_LOSS = 'log'
    SGD_ALPHA = 0.0001
    SGD_PENALTY = 'l2'
    SGD_BATCH_SIZE = 128
    SGD_EPOCHS = 10

    # xgboost
    XGB = False
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
    SAVE_RESULTS = False

    #save training data for experimentation and hyperparameter grid search
    SAVEEXPERIMENTDATA = False
    EXPERIMENTDATAFILE = 'data/experimentdatafile.pkl'

    #vocabulary
    VOCAB_SIZE = 10000
    STEM = False
    FULL_VOCAB_SIZE = 1000
    FULL_VOCAB_TYPE = 'character'
    FULL_CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/|_@#$%^&*~`+-=<>()[]{}"
    FULL_VOCAB_STEM = False
    SEED = 41
    
    HDF5_PATH_TRAIN=None
    HDF5_PATH_TEST=None
    HDF5_SAVE_FREQUENCY=100
    HDF5_USE_EXISTING=False

    USE_CACHE = False



@xp.automain
def run_experiment(directory,
            BOW_APPEND,
            BOW_DIFFERENCE,
            BOW_PRODUCT,
            BOW_COS,
            BOW_TFIDF,
            BOW_BINARY,
            ST_APPEND,
            ST_DIFFERENCE,
            ST_PRODUCT,
            ST_COS,
            LDA_APPEND,
            LDA_DIFFERENCE,
            LDA_PRODUCT,
            LDA_COS,
            LDA_TOPICS,
            W2V_AVG,
            W2V_MAX,
            W2V_MIN,
            W2V_ABS,
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
            SGD,
            SGD_LOSS,
            SGD_ALPHA,
            SGD_PENALTY,
            SGD_EPOCHS,
            SGD_BATCH_SIZE,
            MEM_NET,
            MEM_VOCAB,
            MEM_TYPE,
            MEM_BATCH,
            MEM_EPOCHS,
            MEM_MASK_MODE,
            MEM_EMBED_MODE,
            MEM_ONEHOT_MIN_LEN,
            MEM_ONEHOT_MAX_LEN,
            RESAMPLING,
            NOVEL_RATIO,
            OVERSAMPLING,
            REPLACEMENT,
            SAVE_RESULTS,
            SAVEEXPERIMENTDATA,
            EXPERIMENTDATAFILE,
            VOCAB_SIZE,
            STEM,
            FULL_VOCAB_SIZE,
            FULL_VOCAB_TYPE,
            FULL_CHAR_VOCAB,
            FULL_VOCAB_STEM,
            SEED,
            HDF5_PATH_TRAIN,
            HDF5_PATH_TEST,
            HDF5_SAVE_FREQUENCY,
            HDF5_USE_EXISTING,
            USE_CACHE,
            _run):
    # store default metadata
    USER = os.environ.get('USER', 'unknown user')
    _run.info = { 'user': USER }

    return pythia_main(
        get_args(
            directory,
            BOW_APPEND,
            BOW_DIFFERENCE,
            BOW_PRODUCT,
            BOW_COS,
            BOW_TFIDF,
            BOW_BINARY,
            ST_APPEND,
            ST_DIFFERENCE,
            ST_PRODUCT,
            ST_COS,
            LDA_APPEND,
            LDA_DIFFERENCE,
            LDA_PRODUCT,
            LDA_COS,
            LDA_TOPICS,
            W2V_AVG,
            W2V_MAX,
            W2V_MIN,
            W2V_ABS,
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
            SGD,
            SGD_LOSS,
            SGD_ALPHA,
            SGD_PENALTY,
            SGD_EPOCHS,
            SGD_BATCH_SIZE,
            MEM_NET,
            MEM_VOCAB,
            MEM_TYPE,
            MEM_BATCH,
            MEM_EPOCHS,
            MEM_MASK_MODE,
            MEM_EMBED_MODE,
            MEM_ONEHOT_MIN_LEN,
            MEM_ONEHOT_MAX_LEN,
            RESAMPLING,
            NOVEL_RATIO,
            OVERSAMPLING,
            REPLACEMENT,
            SAVE_RESULTS,
            SAVEEXPERIMENTDATA,
            EXPERIMENTDATAFILE,
            VOCAB_SIZE,
            STEM,
            FULL_VOCAB_SIZE,
            FULL_VOCAB_TYPE,
            FULL_CHAR_VOCAB,
            FULL_VOCAB_STEM,
            SEED,
            HDF5_PATH_TRAIN,
            HDF5_PATH_TEST,
            HDF5_SAVE_FREQUENCY,
            HDF5_USE_EXISTING,
            USE_CACHE)
    )
