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


def main(argv):
    '''
    controls the over-arching implmentation of the algorithms
    '''
    directory, features, algorithms, parameters = argv

    # parsing
    print("parsing json data...", file=sys.stderr)
    clusters, order, data, test_clusters, test_order, test_data, corpusdict = parse_json.main([directory, parameters])

    # resampling
    if 'resampling' in parameters:
        print("resampling...", file=sys.stderr)
        data, clusters, order, corpusdict = sample(data, "novelty", **parameters['resampling'])

    # preprocessing
    print("preprocessing...", file=sys.stderr)
    vocab, encoder_decoder, lda = preprocess.main([features, corpusdict, data])

    # featurization
    print("generating training and testing data...", file=sys.stderr)
    train_data, train_target = data_gen.main([clusters, order, data, features, vocab, encoder_decoder, lda])
    test_data, test_target = data_gen.main([test_clusters, test_order, test_data, features, vocab, encoder_decoder, lda])

    # modeling
    print("running algorithms...", file=sys.stderr)
    if 'log_reg' in algorithms:
        logreg_model = log_reg.main([train_data, train_target, algorithms['log_reg']])
        predicted_labels, perform_results = predict.main([logreg_model, test_data, test_target])
    if 'svm' in algorithms:
        svm_model = svm.main([train_data, train_target, algorithms['svm']])
        predicted_labels, perform_results = predict.main([svm_model, test_data, test_target])
    if 'xgb' in algorithms:
        xgb_model = xgb.main([train_data, train_target, algorithms['xgb']])
        predicted_labels, perform_results = predict.main([xgb_model, test_data, test_target])

    # results
    return perform_results


def parse_args(given_args=None):
    parser = argparse.ArgumentParser(description="predict novelty in a corpus")
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

    featureTuple = namedtuple('features', 'cos_similarity, tfidf_sum, bag_of_words, skipthoughts, lda')
    features = featureTuple(args.cos_similarity, args.tfidf_sum, args.bag_of_words, args.skipthoughts, args.LDA)

    algTuple = namedtuple('algorithms', 'log_reg, svm')
    algorithms = algTuple(args.log_reg, args.svm)

    paramTuple = namedtuple('parameters', 'vocab_size, lda_topics, resampling, novel_ratio, oversampling, replacement')
    parameters = paramTuple(args.vocab_size, args.LDA_topics, args.resampling, args.novel_ratio, args.oversampling, args.replacement)

    if not (args.cos_similarity or args.tfidf_sum or args.bag_of_words or args.skipthoughts or args.LDA):
        parser.exit(status=1, message="Error: pipeline requires at least one feature\n")

    if not (args.log_reg or args.svm):
        parser.exit(status=3, message="Error: pipeline requires at least one algorithm\n")

    return [args.directory, features, algorithms, parameters]


def get_args(
    # DIRECTORY
    directory='/data/stackexchange/anime',

    # FEATURES
    # bag of words
    BOW_APPEND=False,
    BOW_DIFFERENCE=True,
    BOW_PRODUCT=True,
    BOW_COS=True,
    BOW_TFIDF=True,
    BOW_VOCAB=None,

    # skipthoughts
    ST_APPEND=False,
    ST_DIFFERENCE=False,
    ST_PRODUCT=False,
    ST_COS=False,

    # lda
    LDA_APPEND=False,
    LDA_DIFFERENCE=False,
    LDA_PRODUCT=False,
    LDA_COS=False,
    LDA_VOCAB=1000,
    LDA_TOPICS=40,

    # ALGORITHMS
    # logistic regression
    LOG_REG=True,
    LOG_PENALTY='l2',
    LOG_TOL=1e-4,
    LOG_C=1e-4,

    # svm
    SVM=True,
    SVM_C=2000,
    SVM_KERNAL='linear',
    SVM_GAMMA='auto',

    # xgboost
    XGB=True,
    XGB_LEARNRATE=0.1,
    XGB_MAXDEPTH=3,
    XGB_MINCHILDWEIGHT=1,
    XGB_COLSAMPLEBYTREE=1,

    # PARAMETERS
    # resampling
    RESAMPLING=True,
    NOVEL_RATIO=None,
    OVERSAMPLING=False,
    REPLACEMENT=False,

        SEED=None):

    # get features
    bow = None
    st = None
    lda = None

    if BOW_APPEND or BOW_DIFFERENCE or BOW_PRODUCT or BOW_COS or BOW_TFIDF:
        bow = dict()
        if BOW_APPEND:
            bow['append'] = BOW_APPEND
        if BOW_DIFFERENCE:
            bow['difference'] = BOW_DIFFERENCE
        if BOW_PRODUCT:
            bow['product'] = BOW_PRODUCT
        if BOW_COS:
            bow['cos'] = BOW_COS
        if BOW_TFIDF:
            bow['tfidf'] = BOW_TFIDF
        if BOW_VOCAB:
            bow['vocab'] = BOW_VOCAB
    if ST_APPEND or ST_DIFFERENCE or ST_PRODUCT or ST_COS:
        st = dict()
        if ST_APPEND:
            st['append'] = ST_APPEND
        if ST_DIFFERENCE:
            st['difference'] = ST_DIFFERENCE
        if ST_PRODUCT:
            st['product'] = ST_PRODUCT
        if ST_COS:
            st['cos'] = ST_COS
    if LDA_APPEND or LDA_DIFFERENCE or LDA_PRODUCT or LDA_COS:
        lda = dict()
        if LDA_APPEND:
            lda['append'] = LDA_APPEND
        if LDA_DIFFERENCE:
            lda['difference'] = LDA_DIFFERENCE
        if LDA_PRODUCT:
            lda['product'] = LDA_PRODUCT
        if LDA_COS:
            lda['cos'] = LDA_COS
        if LDA_VOCAB:
            lda['vocab'] = LDA_VOCAB
        if LDA_TOPICS:
            lda['topics'] = LDA_TOPICS

    features = dict()
    if bow:
        features['bow'] = bow
    if st:
        features['st'] = st
    if lda:
        features['lda'] = lda

    # get algorithms
    log_reg = None
    svm = None
    xgb = None

    if LOG_REG:
        log_reg = dict()
        if LOG_PENALTY:
            log_reg['log_penalty'] = LOG_PENALTY
        if LOG_TOL:
            log_reg['log_tol'] = LOG_TOL
        if LOG_C:
            log_reg['log_C'] = LOG_C
    if SVM:
        svm = dict()
        if SVM_C:
            svm['svm_C'] = SVM_C
        if SVM_KERNAL:
            svm['svm_kernal'] = SVM_KERNAL
        if SVM_GAMMA:
            svm['svm_gamma'] = SVM_GAMMA
    if XGB:
        xgb = dict()
        if XGB_LEARNRATE:
            xgb['x_learning_rate'] = XGB_LEARNRATE
        if XGB_MAXDEPTH:
            xgb['x_max_depth'] = XGB_MAXDEPTH
        if XGB_COLSAMPLEBYTREE:
            xgb['svm_gamma'] = XGB_COLSAMPLEBYTREE
        if XGB_MINCHILDWEIGHT:
            xgb['svm_gamma'] = XGB_MINCHILDWEIGHT

    algorithms = dict()
    if log_reg:
        algorithms['log_reg'] = log_reg
    if svm:
        algorithms['svm'] = svm
    if xgb:
        algorithms['xgb'] = xgb

    # get parameters
    resampling = None

    if RESAMPLING:
        resampling = dict()
        if NOVEL_RATIO:
            resampling['novelToNotNovelRatio'] = NOVEL_RATIO
        if OVERSAMPLING:
            resampling['over'] = OVERSAMPLING
        if REPLACEMENT:
            resampling['replacement'] = REPLACEMENT

    parameters = dict()
    if RESAMPLING:
        parameters['resampling'] = resampling
    if SEED:
        parameters['seed'] = SEED

    return [directory, features, algorithms, parameters]

if __name__ == '__main__':
    #args = parse_args()
    args = get_args()
    print("Algorithm details and Results:", file=sys.stderr)
    print(main(args), file=sys.stdout)
    sys.exit(0)
