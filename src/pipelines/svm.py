#!/usr/bin/env python
# coding: utf-8
from src.utils import performance_metrics
import numpy as np
import sys
from sklearn import svm
from sklearn.svm import SVC
from collections import namedtuple

def run_model(train_data, train_labels, test_data, test_labels, svm_C=1.0, svm_kernel='rbf', svm_degree=3, svm_gamma='auto', svm_coef0=0.0,
              svm_shrinking=True, svm_probability=False, svm_tol=0.001, svm_cache_size=200, svm_class_weight=None, svm_verbose=False,
              svm_max_iter=-1, svm_decision_function_shape=None, svm_random_state=None, *args, **kwargs):
    '''
    Algorithm which will take in a set of training text and labels to train a bag of words model
    This model is then used with a logistic regression algorithm to predict the labels for a second set of text
    Method modified from code available at:
    https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
    Args:
        train_data_text: Text training set.  Needs to be iterable
        train_labels: Training set labels
        test_data_text: The text to
    Returns:
        pred_labels: The predicted labels as determined by logistic regression
    '''

    #use Logistic Regression to train a model
    svm = SVC(C=svm_C, kernel=svm_kernel, degree=svm_degree, gamma=svm_gamma, coef0=svm_coef0, shrinking=svm_shrinking,
              probability=svm_probability, tol=svm_tol, cache_size=svm_cache_size, class_weight=svm_class_weight,
              verbose=svm_verbose, max_iter=svm_max_iter, decision_function_shape=svm_decision_function_shape, random_state=svm_random_state)

    # we create an instance of Neighbours Classifier and fit the data.
    svm.fit(train_data, train_labels)

    #Now that we have something trained we can check if it is accurate with the test set
    pred_labels = svm.predict(test_data)
    perform_results = performance_metrics.get_perform_metrics(test_labels, pred_labels)

    return pred_labels, perform_results

def main(argv):

    train_data, train_target, test_data, test_target = argv[0], argv[1], argv[2], argv[3]

    if len(argv)>4:
        args_dict = argv[4]
    else:
        args_dict = {}

    print("running support vector machine...",file=sys.stderr)

    predicted_labels, perform_results = run_model(train_data, train_target, test_data, test_target, **args_dict)

    return predicted_labels, perform_results

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: log_reg.py train_data, train_labels, test_data, test_labels\n\nCompute log reg between data (defined in train_data, train_labels, test_data, test_labels)")
    else: main(sys.argv[1:])
