#!/usr/bin/env python
# coding: utf-8
import sys
from sklearn.svm import SVC

def run_model(train_data, train_labels, svm_C=1.0, svm_kernel='rbf', svm_degree=3, svm_gamma='auto', svm_coef0=0.0,
              svm_shrinking=True, svm_probability=False, svm_tol=0.001, svm_cache_size=200, svm_class_weight=None, svm_verbose=False,
              svm_max_iter=-1, svm_decision_function_shape=None, svm_random_state=None, **kwargs):
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
    
    return svm

def main(argv):

    train_data, train_target = argv[0], argv[1]

    if len(argv)>2:
        args_dict = argv[2]
    else:
        args_dict = {}

    print("running support vector machine...",file=sys.stderr)

    svm = run_model(train_data, train_target, **args_dict)

    return svm

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: log_reg.py train_data, train_labels, args_dict\n\nCreates svm classifier from the data")
    else: main(sys.argv[1:])
