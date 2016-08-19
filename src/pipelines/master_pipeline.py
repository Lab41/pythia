#!/usr/bin/env python
'''
Controls the pipeline for Pythia.

This module regulates the features and algorithms used in order to detect novelty, 
then adminstrates the implementation of the given specifications. It requires a 
directory full of JSON files, where each file contains a cluster of documents.
'''
import sys
import argparse
from collections import namedtuple
import numpy as np
from src.pipelines import parse_json, preprocess, data_gen, log_reg, svm, xgb, predict
from src.utils.sampling import sample

def main(argv):
    '''
    controls the over-arching implmentation of the algorithms
    '''
    directory, features, algorithms, parameters = argv
    
    # Create a numpy random state
    random_state = np.random.RandomState(parameters['seed'])

    #parsing
    print("parsing json data...",file=sys.stderr)
    clusters, order, data, test_clusters, test_order, test_data, corpusdict = parse_json.main(directory, parameters)

    #resampling
    if 'resampling' in parameters:
        print("resampling...",file=sys.stderr)
        data, clusters, order, corpusdict = sample(data, "novelty", random_state=random_state, **parameters['resampling'])

    #preprocessing
    print("preprocessing...",file=sys.stderr)
    vocab, encoder_decoder, lda, tf_model = preprocess.main(features, parameters, corpusdict, data)

    #featurization
    print("generating training and testing data...",file=sys.stderr)
    train_data, train_target = data_gen.main([clusters, order, data, features, parameters, vocab, encoder_decoder, lda, tf_model])
    test_data, test_target = data_gen.main([test_clusters, test_order, test_data, features, parameters, vocab, encoder_decoder, lda, tf_model])


    #modeling
    print("running algorithms...",file=sys.stderr)
    if 'log_reg' in algorithms:
        logreg_model = log_reg.main([train_data, train_target, algorithms['log_reg']])
        predicted_labels, perform_results = predict.main([logreg_model, test_data, test_target])
    if 'svm' in algorithms:
        svm_model = svm.main([train_data, train_target, algorithms['svm']])
        predicted_labels, perform_results = predict.main([svm_model, test_data, test_target])
    if 'xgb' in algorithms:
        xgb_model = xgb.main([train_data, train_target, algorithms['xgb']])
        predicted_labels, perform_results = predict.main([xgb_model, test_data, test_target])

    #results
    return perform_results

def get_args(
    #DIRECTORY
    directory = 'data/stackexchange/anime',

    #FEATURES
    #bag of words
    BOW_APPEND = True,
    BOW_DIFFERENCE = False,
    BOW_PRODUCT = True,
    BOW_COS = False,
    BOW_TFIDF = False,

    #skipthoughts
    ST_APPEND = False,
    ST_DIFFERENCE = False,
    ST_PRODUCT = False,
    ST_COS = False,

    #lda
    LDA_APPEND = False,
    LDA_DIFFERENCE = False,
    LDA_PRODUCT = False,
    LDA_COS = False,
    LDA_TOPICS = 40,

    #one-hot CNN layer
    CNN_APPEND = False,
    CNN_DIFFERENCE = False,
    CNN_PRODUCT = False,
    CNN_COS = False,
    #The vocabulary can either be character or word
    #If words, WORDONEHOT_VOCAB will be used as the vocab length
    CNN_VOCAB_TYPE = "character",
    CNN_CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789",

    # wordonehot (will not play nicely with other featurization methods b/c not
    # vector)
    WORDONEHOT = False,
    #WORDONEHOT_DOCLENGTH = None
    WORDONEHOT_VOCAB = 5000,

    #ALGORITHMS
    #logistic regression
    LOG_REG = False,
    LOG_PENALTY = 'l2',
    LOG_TOL = 1e-4,
    LOG_C = 1e-4,

    #svm
    SVM = False,
    SVM_C = 2000,
    SVM_KERNEL = 'linear',
    SVM_GAMMA = 'auto',

    #xgboost
    XGB = False,
    XGB_LEARNRATE = 0.1,
    XGB_MAXDEPTH = 3,
    XGB_MINCHILDWEIGHT = 1,
    XGB_COLSAMPLEBYTREE = 1,

    #PARAMETERS
    #resampling
    RESAMPLING = True,
    NOVEL_RATIO = None,
    OVERSAMPLING = False,
    REPLACEMENT = False,


    #vocabulary
    VOCAB_SIZE = 1000,
    STEM = False,

    SEED = None):

    #get features
    bow = None
    st = None
    lda = None
    wordonehot = None
    cnn = None

    if BOW_APPEND or BOW_DIFFERENCE or BOW_PRODUCT or BOW_COS or BOW_TFIDF:
        bow = dict()
        if BOW_APPEND: bow['append'] = BOW_APPEND
        if BOW_DIFFERENCE: bow['difference'] = BOW_DIFFERENCE
        if BOW_PRODUCT: bow['product'] = BOW_PRODUCT
        if BOW_COS: bow['cos'] = BOW_COS
        if BOW_TFIDF: bow['tfidf'] = BOW_TFIDF
    if ST_APPEND or ST_DIFFERENCE or ST_PRODUCT or ST_COS:
        st = dict()
        if ST_APPEND: st['append'] = ST_APPEND
        if ST_DIFFERENCE: st['difference'] = ST_DIFFERENCE
        if ST_PRODUCT: st['product'] = ST_PRODUCT
        if ST_COS: st['cos'] = ST_COS
    if LDA_APPEND or LDA_DIFFERENCE or LDA_PRODUCT or LDA_COS:
        lda = dict()
        if LDA_APPEND: lda['append'] = LDA_APPEND
        if LDA_DIFFERENCE: lda['difference'] = LDA_DIFFERENCE
        if LDA_PRODUCT: lda['product'] = LDA_PRODUCT
        if LDA_COS: lda['cos'] = LDA_COS
        if LDA_TOPICS: lda['topics'] = LDA_TOPICS
    if WORDONEHOT:
        wordonehot = dict()
        if WORDONEHOT_VOCAB:
            wordonehot['vocab'] = WORDONEHOT_VOCAB
    if CNN_APPEND or CNN_DIFFERENCE or CNN_PRODUCT or CNN_COS:
        cnn = dict()
        if CNN_APPEND: cnn['append'] = CNN_APPEND
        if CNN_DIFFERENCE: cnn['difference'] = CNN_DIFFERENCE
        if CNN_PRODUCT: cnn['product'] = CNN_PRODUCT
        if CNN_COS: cnn['cos'] = CNN_COS
        if CNN_VOCAB_TYPE:
            cnn['vocab_type'] = CNN_VOCAB_TYPE
            if CNN_VOCAB_TYPE=="word":
                if WORDONEHOT_VOCAB: cnn['vocab_len'] = WORDONEHOT_VOCAB
        if CNN_CHAR_VOCAB: cnn['topics'] = CNN_CHAR_VOCAB

    features = dict()
    if bow:
        features['bow'] = bow
    if st:
        features['st'] = st
    if lda:
        features['lda'] = lda
    if wordonehot:
        features['wordonehot'] = wordonehot
    if cnn:
        features['cnn'] = cnn

    #get algorithms
    log_reg = None
    svm = None
    xgb = None

    if LOG_REG:
        log_reg = dict()
        if LOG_PENALTY: log_reg['log_penalty'] = LOG_PENALTY
        if LOG_TOL: log_reg['log_tol'] = LOG_TOL
        if LOG_C: log_reg['log_C'] = LOG_C
    if SVM:
        svm = dict()
        if SVM_C: svm['svm_C'] = SVM_C
        if SVM_KERNAL: svm['svm_kernal'] = SVM_KERNAL
        if SVM_GAMMA: svm['svm_gamma'] = SVM_GAMMA
    if XGB:
        xgb = dict()
        if XGB_LEARNRATE: xgb['x_learning_rate'] = XGB_LEARNRATE
        if XGB_MAXDEPTH: xgb['x_max_depth'] = XGB_MAXDEPTH
        if XGB_COLSAMPLEBYTREE: xgb['svm_gamma'] = XGB_COLSAMPLEBYTREE
        if XGB_MINCHILDWEIGHT: xgb['svm_gamma'] = XGB_MINCHILDWEIGHT

    algorithms = dict()
    if log_reg: algorithms['log_reg'] = log_reg
    if svm: algorithms['svm'] = svm
    if xgb: algorithms['xgb'] = xgb

    #get parameters
    resampling = None

    if RESAMPLING:
        resampling = dict()
        if NOVEL_RATIO: resampling['novelToNotNovelRatio'] = NOVEL_RATIO
        if OVERSAMPLING: resampling['over'] = OVERSAMPLING
        if REPLACEMENT: resampling['replacement'] = REPLACEMENT

    parameters = dict()
    if RESAMPLING: parameters['resampling'] = resampling
    if VOCAB_SIZE: parameters['vocab'] = VOCAB_SIZE
    if STEM: parameters['stem'] = STEM
    if SEED: 
        parameters['seed'] = SEED
    else:
        parameters['seed'] = 41

    return directory, features, algorithms, parameters

if __name__ == '__main__':
    #args = parse_args()
    args = get_args()
    print("Algorithm details and Results:", file=sys.stderr)
    print(main(args), file=sys.stdout)
    sys.exit(0)
