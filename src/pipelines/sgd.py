#!/usr/bin/env python
# coding: utf-8
""" Linear SGD-based models for classification
"""

import sys
import numpy as np
from sklearn import linear_model
from src.utils.hdf5_minibatch import Hdf5BatchIterator

def run_model(train_data_path, data_path="/data", labels_path="/labels", loss='log', penalty='l2', alpha=0.0001, num_epochs=10, seed=None, batch_size=128):
    ''' SGD-based training for classifiers. Allows out-of-core compute
    Args:
        train_data_path: path to HDF5 file on disk
        data_path: path *inside* HDF5 file to data (features dataset)
        labels_path: path *inside* HDF5 file to labels
        loss: 'log'  or 'hinge' for logistic regression or SVM, respectively
        penalty: 'l1', 'l2', or 'elasticnet' (see SGDClassifier docs)
    Returns:
        pred_labels: The predicted labels as determined by logistic regression
    '''

    # use Logistic Regression to train a model
    # we create an instance of SGDClassifier and fit the data.
    sgd_classifier = linear_model.SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, n_iter=1, random_state=seed)
    for epoch_i in range(num_epochs):
        data = Hdf5BatchIterator(train_data_path, data_path, batch_size=batch_size)
        labels = Hdf5BatchIterator(train_data_path, labels_path, batch_size=batch_size)
        for mb_data, mb_labels in zip(data, labels):
            sgd_classifier.partial_fit(mb_data, mb_labels.ravel(), np.array([0, 1]))

    return sgd_classifier

def main(train_data_path, data_path="/data", labels_path="/labels", **kwargs):

    print("running logistic regression with SGD/HDF5...",file=sys.stderr)

    sgd_classifier = run_model(train_data_path, data_path, labels_path, **kwargs)

    return sgd_classifier

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: log_reg.py train_data, train_labels, args_dict\n\nCreates logistic regression classifier from the data")
    else: main(sys.argv[1:])
