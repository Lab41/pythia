#!/usr/bin/env python
'''
Controls the pipeline for Pythia.

This module regulates the features and algorithms used in order to detect novelty, 
then adminstrates the implementation of the given specifications. It requires a 
directory full of JSON files, where each file contains a cluster of documents.
'''
import pdb 
import sys
import os
import pickle
import argparse
import logging
from collections import namedtuple
import numpy as np
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,allow_gc=True"  # Sets flags for use of GPU
from memory_profiler import profile
from src.pipelines import parse_json, preprocess, data_gen, log_reg, svm, xgb, predict, sgd
from src.utils.sampling import sample
from src.mem_net import main_mem_net
from src.utils import hashing
from src.utils.sampling import sample

cache_pickle = "{}.pkl"
cache_dir = ".cache-pythia"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


def main(argv):
    '''
    controls the over-arching implmentation of the algorithms
    '''
    directory, features, algorithms, parameters = argv
    
    # Create a numpy random state
    random_state = np.random.RandomState(parameters['seed'])

    #parsing
    print("parsing json data...",file=sys.stderr)



    if parameters['use_cache']:
        dir_hash = hashing.dir_hash(directory)
        pickle_path = os.path.join(cache_dir, cache_pickle.format(dir_hash))
        try:
            logger.debug("Trying to use cache")
            with open(pickle_path, 'rb') as f:
                parsed_data = pickle.load(f)
                logger.debug("Using existing cache")
        except:
            # parse and write to cache
            logger.debug("Parsing and writing to cache")
            parsed_data = parse_json.main(directory, parameters)
            os.makedirs(cache_dir, exist_ok=True)
            with open(pickle_path, 'wb') as f:
                pickle.dump(parsed_data, f) 
    else:
        parsed_data = parse_json.main(directory, parameters)
    clusters, order, data, test_clusters, test_order, test_data, corpusdict = parsed_data

    #preprocessing
    print("preprocessing...",file=sys.stderr)
    vocab, full_vocab, encoder_decoder, lda_model, tf_model, w2v_model = preprocess.main(features, parameters, corpusdict, data)

    #featurization
    hdf5_path_train=parameters['hdf5_path_train']
    hdf5_path_test=parameters['hdf5_path_test']
    print("generating training data...",file=sys.stderr)
    train_data, train_target, train_ids = data_gen.gen_observations(clusters, order, data, features, parameters, vocab, full_vocab, encoder_decoder, lda_model, tf_model, w2v_model, hdf5_path_train)
    print("generating testing data...",file=sys.stderr)
    test_data, test_target, test_ids = data_gen.gen_observations(test_clusters, test_order, test_data, features, parameters, vocab, full_vocab, encoder_decoder, lda_model, tf_model, w2v_model, hdf5_path_test)

    # save training data for separate experimentation and hyperparameter optimization
    if 'saveexperimentdata' in parameters:
        lunchbox = dict()
        lunchbox['directory'] = directory
        lunchbox['features'] = features
        lunchbox['algorithms'] = algorithms
        lunchbox['parameters'] = parameters
        lunchbox['train_data'] = train_data
        lunchbox['train_target'] = train_target
        lunchbox['test_data'] = test_data
        lunchbox['test_target'] = test_target
        pickle.dump(lunchbox, open(parameters['saveexperimentdata']['experimentdatafile'], "wb"))

    #modeling
    print("running algorithms...",file=sys.stderr)
    if 'log_reg' in algorithms:
        logreg_model = log_reg.main([train_data, train_target, algorithms['log_reg']])
        predicted_labels, perform_results = predict.predicter(logreg_model, test_data, test_target)
    if 'svm' in algorithms:
        svm_model = svm.main([train_data, train_target, algorithms['svm']])
        predicted_labels, perform_results = predict.predicter(svm_model, test_data, test_target)
    if 'xgb' in algorithms:
        xgb_model = xgb.main([train_data, train_target, algorithms['xgb']])
        predicted_labels, perform_results = predict.predicter(xgb_model, test_data, test_target)
    if 'sgd' in algorithms:
        sgd_model = sgd.main(hdf5_path_train, "/data", "/labels", **algorithms['sgd'])
        predicted_labels, perform_results = predict.predicter(sgd_model, test_data, test_target)
    if 'mem_net' in algorithms:
        from src.mem_net import main_mem_net
        mem_net_model, model_name = main_mem_net.run_mem_net(train_data, test_data, corpusdict, **algorithms['mem_net'])
        predicted_labels, perform_results = main_mem_net.test_mem_network(mem_net_model, model_name, **algorithms['mem_net'])
    #results
    if "save_results" in parameters:
        perform_results.update({"id":test_ids})
        perform_results.update({"predicted_label":predicted_labels.tolist()})
        perform_results.update({"novelty":test_target})
    
    return perform_results

def get_args(
    #DIRECTORY
    directory = 'data/stackexchange/anime',

    #FEATURES
    #bag of words
    BOW_APPEND = False,
    BOW_DIFFERENCE = False,
    BOW_PRODUCT = False,
    BOW_COS = False,
    BOW_TFIDF = False,
    BOW_BINARY = True,

    #skipthoughts
    ST_APPEND = False,
    ST_DIFFERENCE = False,
    ST_PRODUCT = False,
    ST_COS = False,

    #lda
    LDA_APPEND = False,
    LDA_DIFFERENCE = False,
    LDA_PRODUCT = False,
    LDA_COS = False,
    LDA_TOPICS = 40,

    #word2vec
    # If AVG, MAX, MIN or ABS are selected, APPEND, DIFFERENCE, PRODUCT or COS must be selected
    W2V_AVG = False,
    W2V_MAX = False,
    W2V_MIN = False,
    W2V_ABS = False,
    # If APPEND, DIFFERENCE, PRODUCT or COS are selected AVG, MAX, MIN or ABS must be selected
    W2V_APPEND = False,
    W2V_DIFFERENCE = False,
    W2V_PRODUCT = False,
    W2V_COS = False,
    W2V_PRETRAINED=False,
    W2V_MIN_COUNT = 5,
    W2V_WINDOW = 5,
    # W2V_SIZE should be set to 300 if using the Google News pretrained word2vec model
    W2V_SIZE = 300,
    W2V_WORKERS = 3,

    #one-hot CNN layer
    CNN_APPEND = False,
    CNN_DIFFERENCE = False,
    CNN_PRODUCT = False,
    CNN_COS = False,
    #The one-hot CNN will use the full_vocab parameters

    # wordonehot (will not play nicely with other featurization methods b/c not
    # vector)
    WORDONEHOT = False,
    #WORDONEHOT_DOCLENGTH = None
    WORDONEHOT_VOCAB = 5000,

    #ALGORITHMS
    #logistic regression
    LOG_REG = False,
    LOG_PENALTY = 'l2',
    LOG_TOL = 1e-4,
    LOG_C = 1e-4,

    #svm
    SVM = False,
    SVM_C = 2000,
    SVM_KERNEL = 'linear',
    SVM_GAMMA = 'auto',

    #xgboost
    XGB = False,
    XGB_LEARNRATE = 0.1,
    XGB_MAXDEPTH = 3,
    XGB_MINCHILDWEIGHT = 1,
    XGB_COLSAMPLEBYTREE = 1,
    
    # SGD Logistic regression
    SGD = False,
    SGD_LOSS = 'log',
    SGD_ALPHA = 0.0001,
    SGD_PENALTY = 'l2',
    SGD_EPOCHS = 10,
    SGD_BATCH_SIZE = 128,

    #memory network
    MEM_NET = False,
    #The memory network vocab uses Glove which can be 50, 100, 200 or 300 depending on the models you have in /data/glove
    MEM_VOCAB = 50,
    MEM_TYPE = 'dmn_basic',
    MEM_BATCH = 1,
    MEM_EPOCHS = 5,
    MEM_MASK_MODE = 'sentence',
    MEM_EMBED_MODE = "word2vec",
    MEM_ONEHOT_MIN_LEN = 140,
    MEM_ONEHOT_MAX_LEN = 1000,

    #PARAMETERS
    #resampling
    RESAMPLING = False,
    NOVEL_RATIO = None,
    OVERSAMPLING = False,
    REPLACEMENT = False,
    SAVE_RESULTS = False,

    #save training data for experimentation and hyperparameter grid search
    SAVEEXPERIMENTDATA = False,
    EXPERIMENTDATAFILE='data/experimentdatafile.pkl',

    #vocabulary
    VOCAB_SIZE = 10000,
    STEM = False,
    FULL_VOCAB_SIZE = 10000,
    FULL_VOCAB_TYPE = 'character',
    FULL_CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/|_@#$%^&*~`+-=<>()[]{}",
    FULL_VOCAB_STEM = False,
    SEED = 41,
    
    HDF5_PATH_TRAIN = None,
    HDF5_PATH_TEST = None,
    HDF5_SAVE_FREQUENCY = 100,
    HDF5_USE_EXISTING = True,
    
    USE_CACHE = False):
    """ Return a parameters data structure with information on how to
    run an experiment. Argument list should match experiments/experiments.py
    """

    #get features
    bow = None
    st = None
    lda = None
    w2v = None
    wordonehot = None
    cnn = None
    mem_net = None

    if BOW_APPEND or BOW_DIFFERENCE or BOW_PRODUCT or BOW_COS or BOW_TFIDF:
        bow = dict()
        if BOW_APPEND: bow['append'] = BOW_APPEND
        if BOW_DIFFERENCE: bow['difference'] = BOW_DIFFERENCE
        if BOW_PRODUCT: bow['product'] = BOW_PRODUCT
        if BOW_COS: bow['cos'] = BOW_COS
        if BOW_TFIDF: bow['tfidf'] = BOW_TFIDF
        if BOW_BINARY: bow['binary'] = BOW_BINARY
    if ST_APPEND or ST_DIFFERENCE or ST_PRODUCT or ST_COS:
        st = dict()
        if ST_APPEND: st['append'] = ST_APPEND
        if ST_DIFFERENCE: st['difference'] = ST_DIFFERENCE
        if ST_PRODUCT: st['product'] = ST_PRODUCT
        if ST_COS: st['cos'] = ST_COS
    if LDA_APPEND or LDA_DIFFERENCE or LDA_PRODUCT or LDA_COS:
        lda = dict()
        if LDA_APPEND: lda['append'] = LDA_APPEND
        if LDA_DIFFERENCE: lda['difference'] = LDA_DIFFERENCE
        if LDA_PRODUCT: lda['product'] = LDA_PRODUCT
        if LDA_COS: lda['cos'] = LDA_COS
        if LDA_TOPICS: lda['topics'] = LDA_TOPICS
    if any([W2V_APPEND,W2V_DIFFERENCE,W2V_PRODUCT,W2V_COS]) or any([W2V_AVG,W2V_MAX,W2V_MIN,W2V_ABS]):
        w2v = dict()
        if W2V_AVG: w2v['avg'] = W2V_AVG
        if W2V_MAX: w2v['max'] = W2V_MAX
        if W2V_MIN: w2v['min'] = W2V_MIN
        if W2V_ABS: w2v['abs'] = W2V_ABS
        if W2V_APPEND: w2v['append'] = W2V_APPEND
        if W2V_DIFFERENCE: w2v['difference'] = W2V_DIFFERENCE
        if W2V_PRODUCT: w2v['product'] = W2V_PRODUCT
        if W2V_COS: w2v['cos'] = W2V_COS
        if W2V_PRETRAINED: w2v['pretrained'] = W2V_PRETRAINED
        if W2V_MIN_COUNT: w2v['min_count'] = W2V_MIN_COUNT
        if W2V_WINDOW: w2v['window'] = W2V_WINDOW
        if W2V_SIZE: w2v['size'] = W2V_SIZE
        if W2V_WORKERS: w2v['workers'] = W2V_WORKERS
    if WORDONEHOT:
        wordonehot = dict()
        if WORDONEHOT_VOCAB:
            wordonehot['vocab'] = WORDONEHOT_VOCAB
    if CNN_APPEND or CNN_DIFFERENCE or CNN_PRODUCT or CNN_COS:
        cnn = dict()
        if CNN_APPEND: cnn['append'] = CNN_APPEND
        if CNN_DIFFERENCE: cnn['difference'] = CNN_DIFFERENCE
        if CNN_PRODUCT: cnn['product'] = CNN_PRODUCT
        if CNN_COS: cnn['cos'] = CNN_COS
    if MEM_NET:
        mem_net = dict()
        if MEM_VOCAB: mem_net['word_vector_size'] = MEM_VOCAB
        if SEED: mem_net['seed'] = SEED
        if MEM_TYPE: mem_net['network'] = MEM_TYPE
        if MEM_BATCH: mem_net['batch_size'] = MEM_BATCH
        if MEM_EPOCHS: mem_net['epochs'] = MEM_EPOCHS
        if MEM_MASK_MODE: mem_net['mask_mode'] = MEM_MASK_MODE
        if MEM_EMBED_MODE : mem_net['embed_mode'] = MEM_EMBED_MODE
        if MEM_ONEHOT_MIN_LEN: mem_net['onehot_min_len'] = MEM_ONEHOT_MIN_LEN
        if MEM_ONEHOT_MAX_LEN: mem_net['onehot_max_len'] = MEM_ONEHOT_MAX_LEN
        #Use the same input params as word2vec
        if W2V_PRETRAINED: mem_net['pretrained'] = W2V_PRETRAINED
        if W2V_MIN_COUNT: mem_net['min_count'] = W2V_MIN_COUNT
        if W2V_WINDOW: mem_net['window'] = W2V_WINDOW
        if W2V_SIZE: mem_net['size'] = W2V_SIZE
        if W2V_WORKERS: mem_net['workers'] = W2V_WORKERS


    features = dict()
    if bow:
        features['bow'] = bow
    if st:
        features['st'] = st
    if lda:
        features['lda'] = lda
    if w2v:
        features['w2v'] = w2v
    if wordonehot:
        features['wordonehot'] = wordonehot
    if cnn:
        features['cnn'] = cnn
    if mem_net:
        if len(features)>0:
            print("Caution!!  Only the memory network feature and algorithm will be ran as they have to run alone")
        features['mem_net'] = mem_net

    if len(features) == 0:
        print("Error: At least one feature (ex: Bag of Words, LDA, etc.) must be requested per run.", file=sys.stderr)
        quit()
    w2v_types = [W2V_AVG,W2V_MAX,W2V_MIN,W2V_ABS]
    w2v_ops = [W2V_APPEND,W2V_DIFFERENCE,W2V_PRODUCT,W2V_COS]
    if any(w2v_ops) and not any(w2v_types):
        print("Caution!!  A Word2Vec vector type must be selected. Default will be set to average (W2V_AVG)", file=sys.stderr)
        features['w2v']['avg'] = True
    if any(w2v_types) and not any(w2v_ops):
        print("Caution!!  A Word2Vec vector operation must be selected. Default will be set to append (W2V_APPEND)", file=sys.stderr)
        features['w2v']['append'] = True

    #get algorithms
    log_reg = None
    svm = None
    xgb = None
    sgd = None

    
    if LOG_REG:
        log_reg = dict()
        if LOG_PENALTY: log_reg['log_penalty'] = LOG_PENALTY
        if LOG_TOL: log_reg['log_tol'] = LOG_TOL
        if LOG_C: log_reg['log_C'] = LOG_C
    if SVM:
        svm = dict()
        if SVM_C: svm['svm_C'] = SVM_C
        if SVM_KERNEL: svm['svm_kernel'] = SVM_KERNEL
        if SVM_GAMMA: svm['svm_gamma'] = SVM_GAMMA
    if XGB:
        xgb = dict()
        if XGB_LEARNRATE: xgb['x_learning_rate'] = XGB_LEARNRATE
        if XGB_MAXDEPTH: xgb['x_max_depth'] = XGB_MAXDEPTH
        if XGB_COLSAMPLEBYTREE: xgb['x_colsample_bytree'] = XGB_COLSAMPLEBYTREE
        if XGB_MINCHILDWEIGHT: xgb['x_colsample_bylevel'] = XGB_MINCHILDWEIGHT
    if SGD:
        sgd = dict()
        sgd['alpha'] = SGD_ALPHA
        sgd['loss'] = SGD_LOSS
        sgd['penalty'] = SGD_PENALTY
        sgd['num_epochs'] = SGD_EPOCHS
        sgd['batch_size'] = SGD_BATCH_SIZE
        sgd['seed'] = SEED
        assert HDF5_PATH_TRAIN is not None, "SGD-based methods should be used with HDF5"

    algorithms = dict()    
    if log_reg: algorithms['log_reg'] = log_reg
    if svm: algorithms['svm'] = svm
    if xgb: algorithms['xgb'] = xgb
    if mem_net:
        algorithms['mem_net']=mem_net
    if sgd:
        algorithms['sgd'] = sgd

    logger.debug("Algorithms structure: {}".format(algorithms))

    # Enforce requirement and limitation of one algorithm per run
    if len(algorithms) == 0:
        print("Error: One classification algorithm must be requested per run.", file=sys.stderr)
        quit()
    elif len(algorithms) > 1:
        print("Error: Only one classification can be requested per run.", file=sys.stderr)
        quit()


    #get parameters
    resampling = None

    if RESAMPLING:
        resampling = dict()
        if NOVEL_RATIO: 
            resampling['novelToNotNovelRatio'] = NOVEL_RATIO
            logger.warn("NOVEL_RATIO specified but not supported")
        resampling['over'] = OVERSAMPLING
        resampling['replacement'] = REPLACEMENT

    saveexperimentdata = None
    if SAVEEXPERIMENTDATA:
        saveexperimentdata = dict()
        if EXPERIMENTDATAFILE: saveexperimentdata['experimentdatafile'] = EXPERIMENTDATAFILE

    parameters = dict()
    if RESAMPLING: parameters['resampling'] = resampling
    if SAVE_RESULTS: parameters['save_results'] = SAVE_RESULTS
    if SAVEEXPERIMENTDATA: parameters['saveexperimentdata'] = saveexperimentdata
    if VOCAB_SIZE: parameters['vocab'] = VOCAB_SIZE
    if STEM: parameters['stem'] = STEM
    if SEED: 
        parameters['seed'] = SEED
    else:
        parameters['seed'] = 41
    if FULL_VOCAB_SIZE: parameters['full_vocab_size'] = FULL_VOCAB_SIZE
    if FULL_VOCAB_TYPE: parameters['full_vocab_type'] = FULL_VOCAB_TYPE
    if FULL_CHAR_VOCAB: parameters['full_char_vocab'] = FULL_CHAR_VOCAB
    if FULL_VOCAB_STEM: parameters['full_vocab_stem'] = FULL_VOCAB_STEM

    assert (HDF5_PATH_TRAIN and SGD) or (not HDF5_PATH_TRAIN and not SGD)
    parameters['hdf5_path_test'] = HDF5_PATH_TEST
    parameters['hdf5_path_train'] = HDF5_PATH_TRAIN
    parameters['hdf5_save_frequency'] = HDF5_SAVE_FREQUENCY
    parameters['hdf5_use_existing'] = HDF5_USE_EXISTING
    parameters['use_cache'] = USE_CACHE

    return directory, features, algorithms, parameters

if __name__ == '__main__':
    args = get_args()
    print("Algorithm details and Results:", file=sys.stderr)
    print(main(args), file=sys.stdout)
    sys.exit(0)
