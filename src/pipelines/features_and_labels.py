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
        if features.cos_similarity:
            feature.append(np.array([score.bagwordsScore]))
        if features.tfidf_sum:
            feature.append(np.array([score.tfidfScore]))
        if features.bag_of_words:
            feature.append(score.bog)
        if features.skipthoughts:
            feature.append(score.skipthoughts)
        if features.lda:
            feature.append(score.ldavector)
        feature = np.concatenate(feature, axis=0)
        data.append(feature)
        if score.novelty:
            labels.append(1)
        else:
            labels.append(0)

    return data, labels


def main(argv):

    data, labels = get_data(argv[0], argv[1])

    return data, labels

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: features_and_labels.py scores features\n\ngenerates features and labels for scores given the defined features")
    else:
        main(sys.argv[1:])
