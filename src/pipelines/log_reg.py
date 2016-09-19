#!/usr/bin/env python
# coding: utf-8
import sys
from sklearn import linear_model

def run_model(train_data, train_labels, log_penalty='l2', log_dual=False, log_tol=1e-4, log_C=1e-4,
              log_fit_intercept=True, log_intercept_scaling=1, log_class_weight=None, log_random_state=None,
              log_solver='liblinear', log_max_iter=100, log_multi_class='ovr', log_verbose=0, log_warm_start=False,
              log_n_jobs=1, **kwargs):
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
    logreg = linear_model.LogisticRegression(penalty=log_penalty, dual=log_dual, tol=log_tol, C=log_C,
                                             fit_intercept=log_fit_intercept,
                                             intercept_scaling=log_intercept_scaling, class_weight=log_class_weight,
                                             random_state=log_random_state, solver=log_solver, max_iter=log_max_iter,
                                             multi_class=log_multi_class, verbose=log_verbose, warm_start=log_warm_start,
                                             n_jobs=log_n_jobs)

    # we create an instance of logistic regression classifier and fit the data.
    logreg.fit(train_data, train_labels)

    return logreg

def main(argv):

    train_data, train_target = argv[0], argv[1]

    if len(argv)>2:
        args_dict = argv[2]
    else:
        args_dict = {}

    print("running logistic regression...",file=sys.stderr)

    logreg = run_model(train_data, train_target, **args_dict)

    return logreg

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: log_reg.py train_data, train_labels, args_dict\n\nCreates logistic regression classifier from the data")
    else: main(sys.argv[1:])
