#!/usr/bin/env python
'''
Controls the pipeline for Pythia.

This module regulates the features and algorithms used in order to detect novelty, then adminstrates the implementation of the given specifications. It requires a directory full of JSON files, where each file contains a cluster of documents.
'''
import sys
import argparse
from collections import namedtuple
from src.pipelines import parse_json, preprocess, observations, features_and_labels, log_reg, svm
from src.utils.sampling import sample

def main(argv):
    '''
    controls the over-arching implmentation of the algorithms
    '''
    directory = argv[0]
    features = argv[1]
    algorithms = argv[2]
    parameters = argv[3]

    #parsing
    print("parsing json data...",file=sys.stderr)
    clusters, order, data, test_clusters, test_order, test_data, corpusdict = parse_json.main([directory])

    #resampling
    if parameters.resampling is True:
        print("resampling...",file=sys.stderr)
        data, clusters, order, corpusdict = sample(data, "novelty", novelToNotNovelRatio = parameters.novel_ratio, over = parameters.oversampling, replacement = parameters.replacement)
        
    #preprocessing
    print("preprocessing...",file=sys.stderr)
    vocab, encoder_decoder, lda = preprocess.main([features, corpusdict, data, parameters])

    #featurization step 1
    print("generating observations and features...",file=sys.stderr)
    train_observations = observations.main([clusters, order, data, directory, features, vocab, encoder_decoder, lda])
    test_observations = observations.main([test_clusters, test_order, test_data, directory, features, vocab, encoder_decoder, lda])

    #featurization step 2
    print("generating training and testing data...",file=sys.stderr)
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

if __name__ == '__main__':
    args = parse_args()
    print("Algorithm details and Results:",file=sys.stderr)
    print(main(args), file=sys.stdout)
    sys.exit(0)
