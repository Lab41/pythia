#!/usr/bin/env python
# coding: utf-8

import numpy as np
from collections import namedtuple


def get_data(scores, features):
    length = 0.8 * len(scores)
    data = list()
    labels = list()

    for score in scores:
        feature = list()
        if features.cosine:
            feature.append(np.array([score.bagwordsScore]))
        if features.tf_idf:
            feature.append(np.array([score.tfidfScore]))
        feature = np.concatenate(feature, axis=0)
        data.append(feature)
        if score.novelty: labels.append(1)
        else: labels.append(0)


    return data, labels

def main(argv):
    
    data, labels = get_data(argv[0], argv[1])
    
    return data, labels
    
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print ("Usage: featues_and_labelspy scores features\n\ngenerates features and labels for scores given the defined features")
    else: main(sys.argv[1:])