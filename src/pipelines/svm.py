#!/usr/bin/env python
# coding: utf-8
from src.utils import performance_metrics
import numpy as np
import sys
from sklearn import svm
from sklearn.svm import SVC
from collections import namedtuple

def run_model(train_data, train_labels, test_data, test_labels):
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
    svm = SVC()
    
    # we create an instance of Neighbours Classifier and fit the data.
    svm.fit(train_data, train_labels)
    
    #Now that we have something trained we can check if it is accurate with the test set
    pred_labels = svm.predict(test_data)
    perform_results = performance_metrics.get_perform_metrics(test_labels, pred_labels)
    
    #Perform_results is a dictionary, so we should add other pertinent information to the run
    perform_results['vector'] = 'Bag_of_Words'
    perform_results['alg'] = 'Support_Vector_Machine'

    return pred_labels, perform_results

def main(argv):
    
    train_data, train_target, test_data, test_target = argv[0], argv[1], argv[2], argv[3]
    
    print("running support vector machine...")

    predicted_labels, perform_results = run_model(train_data, train_target, test_data, test_target)

    return predicted_labels, perform_results
    
if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: log_reg.py train_data, train_labels, test_data, test_labels\n\nCompute log reg between data (defined in train_data, train_labels, test_data, test_labels)")
    else: main(sys.argv[1:])
