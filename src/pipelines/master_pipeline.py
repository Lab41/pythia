#!/usr/bin/env python
'''
Controls the pipeline for Pythia.

This module regulates the features and algorithms used in order to detect novelty, then adminstrates the implementation of the given specifications. It requires a directory full of JSON files, where each file contains a cluster of documents.
'''

import sys
import argparse
from collections import namedtuple
from src.pipelines import parse_json, preprocess, observations, features_and_labels, log_reg, svm

def main(argv):
    '''
    Controls the over-arching implmentation of the algorithms.
    
    First, the JSON files are parsed, and the necessary information is extracted and saved into internal data structures.
    '''

    directory = argv[0]
    features = argv[1]
    algorithms = argv[2]

    #parsing
    print("parsing json data...")
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
    print("running algorithms...")
    if algorithms.log_reg:
        predicted_labels, perform_results = log_reg.main([train_data, train_target, test_data, test_target])
    if algorithms.svm:
        predicted_labels, perform_results = svm.main([train_data, train_target, test_data, test_target])
    #results
    print("Algorithm details and Results:")
    print(perform_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "predict novelty in a corpus")
    parser.add_argument("directory", help="directory holding corpus")
    parser.add_argument("--cos_similarity", "-c", help="add cosine similarity as a feature", action="store_true")
    parser.add_argument("--tfidf_sum", "-t", help="add tfidf sum as a feature", action="store_true")
    parser.add_argument("--bag_of_words", "-b", help="add bag of words vectors as a feature", action="store_true")
    parser.add_argument("--skipthoughts", "-k", help="add skipthought vectors as a feature", action="store_true")
    parser.add_argument("--log_reg", "-l", help="run logistic regression", action="store_true")
    parser.add_argument("--svm", "-s", help="run support vector machine", action="store_true")

    args = parser.parse_args()

    featureTuple = namedtuple('features','cos_similarity, tfidf_sum, bag_of_words, skipthoughts')
    features = featureTuple(args.cos_similarity, args.tfidf_sum, args.bag_of_words, args.skipthoughts)

    algTuple = namedtuple('algorithms','log_reg, svm')
    algorithms = algTuple(args.log_reg, args.svm)

    if not (args.cos_similarity or args.tfidf_sum or args.bag_of_words or args.skipthoughts):
        parser.exit(status=1, message="Error: pipeline requires at least one feature\n")

    if not (args.log_reg or args.svm):
        parser.exit(status=3, message="Error: pipeline requires at least one algorithm\n")

    args = [args.directory, features, algorithms]
    print(args)
    main(args)
    parser.exit(status=0, message=None)
