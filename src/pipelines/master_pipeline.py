#!/usr/bin/env python
'''
Controls the pipeline for Pythia.

This module regulates the features and algorithms used in order to detect novelty, then adminstrates the implementation of the given specifications. It requires a directory full of JSON files, where each file contains a cluster of documents.
'''
import sys
import argparse
from collections import namedtuple
from src.pipelines import parse_json, preprocess, data_gen, log_reg, svm, xgb, predict
from src.utils.sampling import sample

import pickle

def main(argv):
    '''
    controls the over-arching implmentation of the algorithms
    '''
    directory, features, algorithms, parameters = argv

    #parsing
    print("parsing json data...",file=sys.stderr)
    clusters, order, data, test_clusters, test_order, test_data, corpusdict = parse_json.main([directory, parameters])

    #resampling
    if 'resampling' in parameters:
        print("resampling...",file=sys.stderr)
        data, clusters, order, corpusdict = sample(data, "novelty", **parameters['resampling'])

    #preprocessing
    print("preprocessing...",file=sys.stderr)
    vocab, encoder_decoder, lda_model, tf_model, w2v_model = preprocess.main([features, parameters, corpusdict, data])

    #featurization
    print("generating training and testing data...",file=sys.stderr)
    train_data, train_target = data_gen.main([clusters, order, data, features, parameters, vocab, encoder_decoder, lda_model, tf_model, w2v_model])
    test_data, test_target = data_gen.main([test_clusters, test_order, test_data, features, parameters, vocab, encoder_decoder, lda_model, tf_model, w2v_model])

    # save training data for separate experimentation and hyperparameter optimization
    if 'savetrainingdata' in parameters:
        pickle.dump(train_data, open(parameters['savetrainingdata']['datafile'], "wb"))
        pickle.dump(train_target, open(parameters['savetrainingdata']['targetfile'], "wb"))

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

def parse_args(given_args=None):
    parser = argparse.ArgumentParser(description = "predict novelty in a corpus")
    parser.add_argument("directory", help="directory holding corpus")
    parser.add_argument("--cos_similarity", "-c", "--cos-similarity", help="add cosine similarity as a feature", action="store_true")
    parser.add_argument("--tfidf_sum", "-t", "--tfidf-sum", help="add tfidf sum as a feature", action="store_true")
    parser.add_argument("--bag_of_words", "-b", "--bag-of-words", help="add bag of words vectors as a feature", action="store_true")
    parser.add_argument("--vocab_size", "-v", type=int, default=500, help="set size of vocabulary to use with features that utilize bag of words")
    parser.add_argument("--skipthoughts", "-k", help="add skipthought vectors as a feature", action="store_true")
    parser.add_argument("--LDA", "-L", help="add Latent Dirichlet Allocation (LDA) vectors as a feature", action="store_true")
    parser.add_argument("--LDA_topics", "-T", type=int, default=10, help="set number of topics for Latent Dirichlet Allocation (LDA) model (default = 10)")
    parser.add_argument("--log_reg", "-l", "--log-reg", help="run logistic regression", action="store_true")
    parser.add_argument("--svm", "-s", help="run support vector machine", action="store_true")
    parser.add_argument("--resampling", "-r", help="conduct resampling to balance novel/non-novel ratio in training data", action="store_true")
    parser.add_argument("--novel_ratio", "-n", type=float, default=1.0, help="set ratio of novel to non-novel sampling during resampling (0<=novel_ratio<=1.0, default = 1.0)")
    parser.add_argument("--oversampling", "-o", help="allow oversampling during resampling", action="store_true")
    parser.add_argument("--replacement", "-p", help="allow replacement during resampling", action="store_true")

    if given_args is not None:
        args, extra_args = parser.parse_known_args(given_args)
    else:
        args = parser.parse_args()

    featureTuple = namedtuple('features','cos_similarity, tfidf_sum, bag_of_words, skipthoughts, lda')
    features = featureTuple(args.cos_similarity, args.tfidf_sum, args.bag_of_words, args.skipthoughts, args.LDA)

    algTuple = namedtuple('algorithms','log_reg, svm')
    algorithms = algTuple(args.log_reg, args.svm)

    paramTuple = namedtuple('parameters','vocab_size, lda_topics, resampling, novel_ratio, oversampling, replacement')
    parameters = paramTuple(args.vocab_size, args.LDA_topics, args.resampling, args.novel_ratio, args.oversampling, args.replacement)

    if not (args.cos_similarity or args.tfidf_sum or args.bag_of_words or args.skipthoughts or args.LDA):
        parser.exit(status=1, message="Error: pipeline requires at least one feature\n")

    if not (args.log_reg or args.svm):
        parser.exit(status=3, message="Error: pipeline requires at least one algorithm\n")

    return [args.directory, features, algorithms, parameters]

def get_args(
    #DIRECTORY
    directory = 'data/stackexchange/anime',

    #FEATURES
    #bag of words
    BOW_APPEND = False,
    BOW_DIFFERENCE = False,
    BOW_PRODUCT = False,
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

    #word2vec
    W2V_APPEND = False,
    W2V_DIFFERENCE = False,
    W2V_PRODUCT = False,
    W2V_COS = False,
    W2V_PRETRAINED=False,
    W2V_MIN_COUNT = 5,
    W2V_WINDOW = 5,
    W2V_SIZE = 100,
    W2V_WORKERS = 3,

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

    #save training data for experimentation and hyperparameter grid search
    SAVETRAININGDATA = False,
    SAVEDATAFILE='data/datafile.pkl',
    SAVETARGETFILE='data/targetfile.pkl',

#vocabulary
    VOCAB_SIZE = 1000,
    STEM = False,

    SEED = None):

    #get features
    bow = None
    st = None
    lda = None
    w2v = None
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
    if W2V_APPEND or W2V_DIFFERENCE or W2V_PRODUCT or W2V_COS:
        w2v = dict()
        if W2V_APPEND: w2v['append'] = W2V_APPEND
        if W2V_DIFFERENCE: w2v['difference'] = W2V_DIFFERENCE
        if W2V_PRODUCT: w2v['product'] = W2V_PRODUCT
        if W2V_COS: w2v['cos'] = W2V_COS
        if W2V_PRETRAINED: w2v['pretrained'] = W2V_PRETRAINED
        if W2V_MIN_COUNT: w2v['min_count'] = W2V_MIN_COUNT
        if W2V_WINDOW: w2v['window'] = W2V_WINDOW
        if W2V_SIZE: w2v['size'] = W2V_SIZE
        if W2V_WORKERS: w2v['workers'] = W2V_WORKERS
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
    if w2v:
        features['w2v'] = w2v
    if wordonehot:
        features['wordonehot'] = wordonehot
    if cnn:
        features['cnn'] = cnn

    if len(features) == 0:
        print("Error: At least one feature (ex: Bag of Words, LDA, etc.) must be requested per run.", file=sys.stderr)
        quit()

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
        if SVM_KERNEL: svm['svm_kernel'] = SVM_KERNEL
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

    # Enforce requirement and limitation of one algorithm per run
    if len(algorithms) == 0:
        print("Error: An algorithm (LOG_REG, SVM, XGB) must be requested per run.", file=sys.stderr)
        quit()
    elif len(algorithms) > 1:
        print("Error: Only one algorithm (LOG_REG, SVM, XGB) can be requested per run.", file=sys.stderr)
        quit()

    #get parameters
    resampling = None

    if RESAMPLING:
        resampling = dict()
        if NOVEL_RATIO: resampling['novelToNotNovelRatio'] = NOVEL_RATIO
        if OVERSAMPLING: resampling['over'] = OVERSAMPLING
        if REPLACEMENT: resampling['replacement'] = REPLACEMENT

    savetrainingdata = None
    if SAVETRAININGDATA:
        savetrainingdata = dict()
        if SAVEDATAFILE: savetrainingdata['datafile'] = SAVEDATAFILE
        if SAVETARGETFILE: savetrainingdata['targetfile'] = SAVETARGETFILE

    parameters = dict()
    if RESAMPLING: parameters['resampling'] = resampling
    if SAVETRAININGDATA: parameters['savetrainingdata'] = savetrainingdata
    if VOCAB_SIZE: parameters['vocab'] = VOCAB_SIZE
    if STEM: parameters['stem'] = STEM
    if SEED: parameters['seed'] = SEED

    return directory, features, algorithms, parameters

if __name__ == '__main__':
    #args = parse_args()
    args = get_args()
    print("Algorithm details and Results:", file=sys.stderr)
    print(main(args), file=sys.stdout)
    sys.exit(0)
