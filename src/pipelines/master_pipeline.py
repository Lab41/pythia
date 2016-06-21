#!/usr/bin/env python

import argparse
from collections import namedtuple
import parse_json, preprocess, observations, features_and_labels, log_reg

def main(argv):
    
    ''' 
    controls the over-arching implmentation of the algorithms
    '''
    
    directory = argv[0]
    features = argv[1]
    
    #parsing
    print("parsing json data...")
    clusters, order, data, test_clusters, test_order, test_data, corpusdict = parse_json.main([directory])
    
    #preprocessing
    vocab = preprocess.main([features, corpusdict])
    
    #featurization step 1
    print("generating observations and features...")
    train_scores = observations.main([clusters, order, data, directory, features, vocab])
    test_scores = observations.main([test_clusters, test_order, test_data, directory, features, vocab])
    
    #featurization step 2
    print("generating training and testing data...")
    train_data, train_target = features_and_labels.main([train_scores, features])
    test_data, test_target = features_and_labels.main([test_scores, features])

    #modeling
    print("running algorithms...")
    predicted_labels, perform_results = log_reg.main([train_data, train_target, test_data, test_target])

    #results
    print("Algorithm details and Results:")
    print(perform_results)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "predict novelty in a corpus")
    parser.add_argument("directory", help="directory holding corpus")
    parser.add_argument("--cosine", "-c", help="add cosine similarity as a feature", action="store_true")
    parser.add_argument("--tf_idf", "-t", help="add tf_idf as a feature", action="store_true")

    args = parser.parse_args()
    
    featureTuple = namedtuple('features','cosine, tf_idf')
    features = featureTuple(args.cosine, args.tf_idf)
    
    if not (args.cosine and args.tf_idf):
        print("Error: pipeline requires at least one feature")
        quit()
    
    args = [args.directory, features]
    print(args)
    main(args)