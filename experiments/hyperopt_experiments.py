import sys
import argparse
from collections import namedtuple
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,allow_gc=True,lib.cnmem=0"  # Sets flags for use of GPU
from src.pipelines import parse_json, preprocess, data_gen, log_reg, svm, xgb, predict, master_pipeline
from src.utils.sampling import sample
from src.mem_net import main_mem_net
import pickle
import time
import src.pipelines.master_pipeline as mp
import numpy as np
from src.pipelines.master_pipeline import main as pythia_main
from sklearn.metrics import precision_recall_fscore_support

import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

directory_base = '/data/stackexchange/anime'
features = {}
algorithms = {}
parameters = {}
parameters['resampling'] = True
parameters['vocab'] = 1000
parameters['stem'] = False
parameters['seed'] = True

log_reg_d = dict()
log_reg_d['log_penalty'] = 'l2'
log_reg_d['log_tol'] = 1e-4
log_reg_d['log_C'] = 1e-4

svm_d = dict()
svm_d['svm_C'] = 2000
svm_d['svm_kernel'] = 'linear'
svm_d['svm_gamma'] = 'auto'

xgb_d = dict()
xgb_d['x_learning_rate'] = 0.1
xgb_d['x_max_depth'] = 3

def objective(args):

    try:
        directory, features, algorithm_type, parameters = args_to_dicts(args)
    #     try:
    #         directory, features, algorithm_type, parameters = args_to_dict(args)
    #     except:
    #         return -1

        print(algorithm_type)
        print(features)

        #parsing
        print("parsing json data...",file=sys.stderr)
        clusters, order, data, test_clusters, test_order, test_data, corpusdict = parse_json.main(directory, parameters)

        #preprocessing
        print("preprocessing...",file=sys.stderr)
        vocab, full_vocab, encoder_decoder, lda_model, tf_model, w2v_model = preprocess.main(features, parameters, corpusdict, data)

        #featurization
        print("generating training and testing data...",file=sys.stderr)
        train_data, train_target = data_gen.main([clusters, order, data, features, parameters, vocab, full_vocab, encoder_decoder, lda_model, tf_model, w2v_model])
        test_data, test_target = data_gen.main([test_clusters, test_order, test_data, features, parameters, vocab, full_vocab, encoder_decoder, lda_model, tf_model, w2v_model])

        if 'log_reg' in algorithm_type:
            model = log_reg.main([train_data, train_target, algorithm_type['log_reg']])
        if 'svm' in algorithm_type:
            model = svm.main([train_data, train_target, algorithm_type['svm']])
        if 'xgb' in algorithm_type:
            model = xgb.main([train_data, train_target, algorithm_type['xgb']])

        predicted_labels, perform_results = predict.main([model, test_data, test_target])

        prfs = precision_recall_fscore_support(test_target, predicted_labels)
        f_score = prfs[2].tolist()
        print(np.mean(f_score))

        return np.mean(f_score)

    except:
        raise

def args_to_dicts(args):

    algorithm_type= args['algorithm_type']
    print(algorithm_type)
    algorithms = {}
    _LOG_REG = False
    _XGB = False
    _SVM = False
    if algorithm_type=='logreg':
        _LOG_REG = True
        algorithms['log_reg'] = log_reg_d
    elif algorithm_type== 'svm':
        _SVM = True
        algorithms['svm'] = svm_d
    else:
        _XGB = True
        algorithms['xgb'] = xgb_d


    args.pop('algorithm_type')
    args['RESAMPLING']=True
    args['USE_CACHE']=True
    args['W2V_PRETRAINED']=True
    print(_LOG_REG, _SVM, _XGB)
    directory, features, algorithm_type, parameters = mp.get_args(
            directory=directory_base, LOG_REG=_LOG_REG, SVM=_SVM, XGB=_XGB, **args)

    return directory, features, algorithm_type, parameters


def run_pythia_hyperopt():

    space = {
        "algorithm_type":hp.choice('algorithm_type', [
                {
                    'type': 'log_reg',
                    'log_C': hp.choice('log_C', [1e-5, 1e-4, 1e-3, 1e-2, 1, 10]),
                    'log_tol': hp.choice('log_tol', [1e-5, 1e-4, 1e-3, 1e-2, 1, 10])
                }, {
                    'type':'svm',
                    'svm_C': hp.choice('svm_C', [2000, 1000])
                }, {
                    'type': 'xgb',
                    'x_learning_rate': hp.choice('x_learning_rate', [0.01, 0.1, 0.5, 1]),
                    'x_max_depth': hp.choice('x_max_depth',[3,4,5,6])
                } ]),
    #     if algorithm_type=='logreg': LOG_REG=True
    #     elif algorithm_type=='svm': SVM=True
    #     else: XGB=True

        "BOW_APPEND":hp.choice('BOW_APPEND', [True, False]),
        "BOW_DIFFERENCE":hp.choice('BOW_DIFFERENCE', [True, False]),
        "BOW_PRODUCT":hp.choice('BOW_PRODUCT', [True, False]),
        "BOW_COS":hp.choice('BOW_COS', [True, False]),
        "BOW_TFIDF":hp.choice('BOW_TFIDF', [True, False]),
    #     "ST_APPEND":hp.choice('ST_APPEND', [True, False]),
    #     "ST_DIFFERENCE":hp.choice('ST_DIFFERENCE', [True, False]),
    #     "ST_PRODUCT":hp.choice('ST_PRODUCT', [True, False]),
    #     "ST_COS":hp.choice('ST_COS', [True, False]),
        "LDA_APPEND":hp.choice('LDA_APPEND', [True, False]),
        "LDA_DIFFERENCE":hp.choice('LDA_DIFFERENCE', [True, False]),
        "LDA_PRODUCT":hp.choice('LDA_PRODUCT', [True, False]),
        "LDA_COS":hp.choice('LDA_COS', [True, False]),
        "LDA_TOPICS":hp.choice('LDA_TOPICS', [True, False]),
    #     "W2V_APPEND":hp.choice('W2V_APPEND', [True, False]),
    #     "W2V_DIFFERENCE":hp.choice('W2V_DIFFERENCE', [True, False]),
    #     "W2V_PRODUCT":hp.choice('W2V_PRODUCT', [True, False]),
    #     "W2V_COS":hp.choice('W2V_COS', [True, False]),
    #     "W2V_PRETRAINED",
    #     "W2V_MIN_COUNT",
    #     "W2V_WINDOW",
    #     "W2V_SIZE",
    #     "W2V_WORKERS",
    #     "CNN_APPEND":hp.choice('CNN_APPEND', [True, False]),
    #     "CNN_DIFFERENCE":hp.choice('CNN_DIFFERENCE', [True, False]),
    #     "CNN_PRODUCT":hp.choice('CNN_PRODUCT', [True, False]),
    #     "CNN_COS":hp.choice('CNN_COS', [True, False]),
    #     "WORDONEHOT",
    #     "WORDONEHOT_VOCAB",
    #     "LOG_REG",
    #     "LOG_PENALTY",
    #     "LOG_TOL",
    #     "LOG_C",
    #     "SVM",
    #     "SVM_C",
    #     "SVM_KERNEL",
    #     "SVM_GAMMA",
    #     "XGB",
    #     "XGB_LEARNRATE",
    #     "XGB_MAXDEPTH",
    #     "XGB_MINCHILDWEIGHT",
    #     'XGB_COLSAMPLEBYTREE',
    #     "RESAMPLING",
    #     "NOVEL_RATIO",
    #     "OVERSAMPLING",
    #     "REPLACEMENT",
    #     "SAVEEXPERIMENTDATA",
    #     "EXPERIMENTDATAFILE",
    #     "VOCAB_SIZE",
    #     "STEM",
    #     "FULL_VOCAB_SIZE",
    #     "FULL_VOCAB_TYPE",
    #     "FULL_CHAR_VOCAB",
    #     "SEED",
    #     'USE_CACHE'
    }
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials = trials)
    print("Best run ", best)
    return trials, best

if __name__ == '__main__':

    trial_results, best = run_pythia_hyperopt()
    with open( "pythia_hyperopt_results" + '.pkl', 'wb') as f:
        pickle.dump(trial_results, f, pickle.HIGHEST_PROTOCOL)