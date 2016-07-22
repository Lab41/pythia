#!/usr/bin/env python
# coding: utf-8
import sys
import xgboost

def run_model(train_data, train_labels, x_max_depth=3, x_learning_rate=0.1, x_n_estimators=100, x_silent=True, x_objective='binary:logistic', x_nthread=-1, x_gamma=0, x_min_child_weight=1, x_max_delta_step=0, x_subsample=1, x_colsample_bytree=1, x_colsample_bylevel=1, x_reg_alpha=0, x_reg_lambda=1, x_scale_pos_weight=1, x_base_score=0.5, x_seed=0, x_missing=None, **kwargs):
    '''
    Takes in a set of training text and labels to fit a XGBoost classifier model. See https://github.com/dmlc/xgboost.
    
    Args:
        train_data: Training set.  Needs to be iterable
        train_labels: Training set labels
    Returns:
        classifier: The XGBoost classifier fit to the training data
    '''

    # Set up xgboost classifier 
    # Parameter information available at:
    # https://xgboost.readthedocs.io/en/latest//python/python_api.html
    # https://xgboost.readthedocs.io/en/latest//parameter.html (Learning Task objectives)
    xclassifier = xgboost.XGBClassifier(max_depth=x_max_depth, learning_rate=x_learning_rate, n_estimators=x_n_estimators, silent=x_silent, objective=x_objective, nthread=x_nthread, gamma=x_gamma, min_child_weight=x_min_child_weight, max_delta_step=x_max_delta_step, subsample=x_subsample, colsample_bytree=x_colsample_bytree, colsample_bylevel=x_colsample_bylevel, reg_alpha=x_reg_alpha, reg_lambda=x_reg_lambda, scale_pos_weight=x_scale_pos_weight, base_score=x_base_score, seed=x_seed, missing=x_missing)

    # Fit XGBoost classifier to training data
    xclassifier.fit(train_data, train_labels)

    # Return XGBoost classifier
    return xclassifier

def main(argv):

    train_data, train_target = argv[0], argv[1]

    if len(argv)>2:
        args_dict = argv[2]
    else:
        args_dict = {}

    print("running xgboost...",file=sys.stderr)

    xclassifier = run_model(train_data, train_target, **args_dict)

    return xclassifier

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: xgb.py train_data, train_labels, args_dict\n\nCreates XGBoost classifier from the data")
    else: main(sys.argv[1:])
