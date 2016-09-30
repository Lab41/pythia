# Running experiments with Pythia

An experiment is an instance of training and evaluating a machine learning model 
on Pythia's novelty detection problem. We have designed a number of 

## experiments.py

Will run a single experiment and optionally log it to a MongoDB instance with Sacred

## do-experiments.sh

Runs experiments specified by a folder of `.json` files, and logs them using Sacred 
to a specified MongoDB instance

## hyperopt_experiments.py

Uses the hyperparameter optimization package hyperopt to design and run experiments, log
them using Sacred, and reveal the best result found

## conduct_auto_sklearn.py

Will run experiments using the package auto-sklearn to find an optimal 
hyperparameter configuration

## conduct_grid_search.py

Uses scikit-learn's GridSearchCV to optimize over a given hyperparameter space
