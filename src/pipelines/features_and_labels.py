#!/usr/bin/env python
# coding: utf-8

import numpy as np
from collections import namedtuple


def get_data(scores, features):
    print "size of training data: ", len(scores)
    length = 0.8 * len(scores)
    #print "size of training data: ", length
    index = 0
    train_data = list()
    train_labels = list()
    test_data = list()
    test_labels = list()
    for score in scores:
        if index < length:
            feature = list()
            if features.cosine:
                feature.append(np.array([score.bagwordsScore]))
            if features.tf_idf:
                feature.append(np.array([score.tfidfScore]))
            feature = np.concatenate(feature, axis=0)
            train_data.append(feature)
            if score.novelty: train_labels.append(1)
            else: train_labels.append(0)
        else:
            feature = list()
            if features.cosine:
                feature.append(np.array([score.bagwordsScore]))
            if features.tf_idf:
                feature.append(np.array([score.tfidfScore]))
            feature = np.concatenate(feature, axis=0)
            test_data.append(feature)
            if score.novelty: test_labels.append(1)
            else: test_labels.append(0)            
        index+=1

    return train_data, train_labels, test_data, test_labels

def main(argv):
    
    train_data, train_target, test_data, test_target = get_data(argv[0], argv[1])
    
    return train_data, train_target, test_data, test_target
    
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: featues_and_labelspy scores features\n\nCompute bag of words log reg between documents defined in JSON file (file1)"
    else: main(sys.argv[1:])