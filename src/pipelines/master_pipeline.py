#!/usr/bin/env python

import argparse
from collections import namedtuple
import parse_json, observations, features_and_labels, log_reg

def main(argv):
    
    ''' 
    controls the over-arching implmentation of the algorithms
    '''
    
    directory = argv[0]
    features = argv[1]
    
    print "parsing json data..."
    allClusters, lookupOrder, documentData, corpusDict = parse_json.main([directory])
    
    
    print "generating observations and features..."
    scores = observations.main([allClusters, lookupOrder, documentData, directory, features, corpusDict])
    
    print "generating training and testing data..."
    train_data, train_target, test_data, test_target = features_and_labels.main([scores, features])

    print "running algorithms..."
    predicted_labels, perform_results = log_reg.main([train_data, train_target, test_data, test_target])

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
    
    args = [args.directory, features]
    print args
    
    main(args)

    # if len(sys.argv) < 3:
    #     print "Usage: master_pipeline.py dir1 feature1 feature2...\n\n\computes logistic regression on data in directory (dir1) using features (feature1, featrue2...)"
    # else: main(sys.argv[1:])