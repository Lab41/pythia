#!/usr/bin/env python
'''
Controls the pipeline for Pythia.

This module regulates the features and algorithms used in order to detect novelty, then adminstrates the implementation of the given specifications. It requires a directory full of JSON files, where each file contains a cluster of documents.
'''

import sys
import json
import argparse
from collections import namedtuple
from src.pipelines import parse_json, preprocess, observations, features_and_labels, log_reg, svm

def main(argv):
    '''
    controls the over-arching implmentation of the algorithms
    '''
    directory = argv[0]
    features = argv[1]
    algorithms = argv[2]

    #parsing
    print("parsing json data...",file=sys.stderr)
    clusters, order, data, test_clusters, test_order, test_data, corpusdict = parse_json.main([directory])

    #preprocessing
    vocab, encoder_decoder = preprocess.main([features, corpusdict])

    #featurization step 1
    print("generating observations and features...")
    train_observations = observations.main([clusters, order, data, directory, features, vocab, encoder_decoder])
    test_observations = observations.main([test_clusters, test_order, test_data, directory, features, vocab, encoder_decoder])

    #featurization step 2
    print("generating training and testing data...")
    train_data, train_target = features_and_labels.main([train_observations, features])
    test_data, test_target = features_and_labels.main([test_observations, features])

    #modeling
    print("running algorithms...",file=sys.stderr)
    if algorithms.log_reg:
        predicted_labels, perform_results = log_reg.main([train_data, train_target, test_data, test_target])
    if algorithms.svm:
        predicted_labels, perform_results = svm.main([train_data, train_target, test_data, test_target])
    #results
    return perform_results

def parse_args(given_args=None):
    parser = argparse.ArgumentParser(description = "predict novelty in a corpus")
    parser.add_argument("directory", help="directory holding corpus")
    parser.add_argument("--cosine", "-c", help="add cosine similarity as a feature", action="store_true")
    parser.add_argument("--tf_idf", "-t", help="add tf_idf as a feature", action="store_true")
    parser.add_argument("--bag_of_words", "-b", "--bag-of-words", help="add bag of words vectors as a feature", action="store_true")
    parser.add_argument("--log_reg", "-l", "--log-reg", help="run logistic regression", action="store_true")
    parser.add_argument("--svm", "-s", help="run support vector machine", action="store_true")

    if given_args is not None:
        args, extra_args = parser.parse_known_args(given_args)
    else:
        args = parser.parse_args()

    featureTuple = namedtuple('features','cosine, tf_idf, bog')
    features = featureTuple(args.cosine, args.tf_idf, args.bag_of_words)

    algTuple = namedtuple('algorithms','log_reg, svm')
    algorithms = algTuple(args.log_reg, args.svm)
    if not (args.cosine or args.tf_idf or args.bag_of_words):
        parser.exit(status=1, message="Error: pipeline requires at least one feature\n")

    if not (args.log_reg or args.svm):
        parser.exit(status=3, message="Error: pipeline requires at least one algorithm\n")

    return [args.directory, features, algorithms]

if __name__ == '__main__':
    args = parse_args()
    print("Algorithm details and Results:",file=sys.stderr)
    print(main(args), file=sys.stdout)
    sys.exit(0)
