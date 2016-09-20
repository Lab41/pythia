# This is the main class for the Dynamic Memory Networks implementation modified to take folders for the input train and test.

import sys
import numpy as np
import sklearn.metrics as metrics
import argparse
import time
import json
from src.utils import performance_metrics

from src.mem_net import utils
from src.mem_net import nn_utils
from src.mem_net import dmn_basic, dmn_basic_w_cnn, dmn_batch

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"  # Sets flags for use of GPU
import errno

#-----------------------------------------------------------------------------#
# Look at environment variable 'PYTHIA_MODELS_PATH' for user-defined model location
# If environment variable is not defined, use current working directory
#-----------------------------------------------------------------------------#
if os.environ.get('PYTHIA_MODELS_PATH') is not None:
    path_to_models = os.path.join(os.environ.get('PYTHIA_MODELS_PATH'), 'mem_net_states')
else:
    path_to_models = os.path.join(os.getcwd(), 'mem_net_states')
#-----------------------------------------------------------------------------#

def parse_args(given_args=None):

    print("==> parsing input arguments")
    parser = argparse.ArgumentParser()

    parser.add_argument('--network', type=str, default="dmn_batch", help='network type: dmn_basic, dmn_smooth, or dmn_batch')
    parser.add_argument('--word_vector_size', type=int, default=50, help='embeding size (50, 100, 200, 300 only)')
    parser.add_argument('--dim', type=int, default=40, help='number of hidden units in input module GRU')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--load_state', type=str, default="", help='state file path')
    parser.add_argument('--answer_module', type=str, default="feedforward", help='answer module type: feedforward or recurrent')
    parser.add_argument('--mode', type=str, default="train", help='mode: train or test. Test mode required load_state')
    parser.add_argument('--input_mask_mode', type=str, default="word", help='input_mask_mode: word or sentence')
    parser.add_argument('--memory_hops', type=int, default=5, help='memory GRU steps')
    parser.add_argument('--batch_size', type=int, default=10, help='no commment')
    parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
    parser.add_argument('--normalize_attention', type=bool, default=False, help='flag for enabling softmax on attention vector')
    # Set arguments for input train and test folders
    parser.add_argument('--input_train', type = str, default=None, help='input_train: folder of data to train with')
    parser.add_argument('--input_test', type = str, default = None, help = 'input_test: folder of data to test with')
    parser.add_argument('--log_every', type=int, default=1, help='print information every x iteration')
    parser.add_argument('--save_every', type=int, default=1, help='save state every x epoch')
    parser.add_argument('--prefix', type=str, default="", help='optional prefix of network name')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate (between 0 and 1)')
    parser.add_argument('--batch_norm', type=bool, default=False, help='batch normalization')
    parser.set_defaults(shuffle=False)
    args = parser.parse_args()

    print(args)

    assert args.word_vector_size in [50, 100, 200, 300]

    network_name = args.prefix + '%s.mh%d.n%d.bs%d%s%s%s.babi%s' % (
        args.network,
        args.memory_hops,
        args.dim,
        args.batch_size,
        ".na" if args.normalize_attention else "",
        ".bn" if args.batch_norm else "",
        (".d" + str(args.dropout)) if args.dropout>0 else "",
        args.input_train.split("/")[-1])

def run_mem_net(train_data, test_data, seed=1, word_vector_size=50,
                network='dmn_basic', batch_size=10, epochs=10, vector_type="word2vec",
                shuffle=False, log_every=100, save_every=2, network_name_pre = '', memory_hops=5, dim=40,
                normalize_attention=False, batch_norm=False,dropout=0.0,
                answer_module = 'feedforward', input_mask_mode = 'sentence', l2 = 0, load_state="", **kwargs):

    # Initialize word2vec with utils.load_glove
    #word2vec = utils.load_glove(word_vector_size)
    # Model likely doesn't need word2vec now as it is used previously
    word2vec = {}
    args_dict = kwargs
    args_dict['word2vec'] = word2vec

    # Go and get the data from the directory folder; see utils class.
    #train_raw, test_raw = utils.get_raw_data(directory, seed, word2vec, word_vector_size)

    #Get the data from the information passed in by the main pythia project
    #This way the same splits and resampling can be used
    # train_raw = utils.analyze_clusters(clusters, order, data)
    # test_raw = utils.analyze_clusters(test_clusters, test_order, test_data)

    args_dict['train_raw'] = train_data
    args_dict['test_raw'] = test_data
    #args to use when the cnn is working
    args_dict['char_vocab'] = list("abcdefghijklmnopqrstuvwxyz0123456789")
    args_dict['vocab_len'] = len(args_dict['char_vocab'])
    args_dict['num_cnn_layers'] = 20
    args_dict['maximum_doc_len'] = 200


    args_dict['word_vector_size']=word_vector_size
    args_dict['dim']=dim
    args_dict['mode']="train"
    args_dict['answer_module'] = answer_module
    args_dict['input_mask_mode'] = input_mask_mode
    args_dict['memory_hops'] = memory_hops
    args_dict['l2'] = l2
    args_dict['normalize_attention'] = normalize_attention
    args_dict['batch_size'] = batch_size
    args_dict['batch_norm'] = batch_norm
    args_dict['dropout'] = dropout

    network_name = network_name_pre + '%s.mh%d.n%d.bs%d%s%s%s' % (
        network,
        memory_hops,
        dim,
        batch_size,
        ".na" if normalize_attention else "",
        ".bn" if batch_norm else "",
        (".d" + str(dropout)) if dropout>0 else "")

    # Check if 'models' directory exists and contains the Skip-Thought models, download if not found
    try:
        os.makedirs(path_to_models)
    except OSError as exception:
        if exception.errno != errno.EEXIST: raise


    # init class
    #TODO try out the classes besides basic and add in if they work
    # Currently only basic works, but hopefully that will change in the future
    # if network == 'dmn_batch':
    #     dmn = dmn_batch.DMN_batch(**args_dict)

    # The basic module is implemented for document similarity
    if network == 'dmn_basic':
        print("Using network " + network)
        if (batch_size != 1):
            print("==> no minibatch training, argument batch_size is useless")
            batch_size = 1
        dmn = dmn_basic.DMN_basic(**args_dict) # Initialize the dmn basic with all the arguments available. This also initializes theano functions and parameters.

    # elif network == 'dmn_cnn':
    #     #raise Exception("Sorry - this isn't working right now!!")
    #
    #     if (batch_size != 1):
    #         print("==> no minibatch training, argument batch_size is useless")
    #         batch_size = 1
    #     dmn = dmn_basic_w_cnn.DMN_basic(**args_dict)
    #
    # elif args.network == 'dmn_smooth':
    #     import dmn_smooth
    #     if (args.batch_size != 1):
    #         print("==> no minibatch training, argument batch_size is useless")
    #         args.batch_size = 1
    #     dmn = dmn_smooth.DMN_smooth(**args_dict)
    #
    # elif args.network == 'dmn_qa':
    #     import dmn_qa_draft
    #     if (args.batch_size != 1):
    #         print("==> no minibatch training, argument batch_size is useless")
    #         args.batch_size = 1
    #     dmn = dmn_qa_draft.DMN_qa(**args_dict)
    #
    else:
        raise Exception("No such valid network known: " + network + " Currently only dmn_basic")


    if load_state != "":
        try:
            dmn.load_state(load_state)
        except:
            pass


    print("==> training"   )
    skipped = 0
    for epoch in range(epochs):
        start_time = time.time()

        if shuffle:
            dmn.shuffle_train_set()

        avg_loss, skipped, perform_results, y_pred = do_epoch(dmn, 'train', epoch, batch_size, log_every) # Run do_epoch for train

        epoch_loss, skipped, perform_results, y_pred = do_epoch(dmn, 'test', epoch, batch_size, log_every, skipped) # Run do_epoch for test

        state_name = path_to_models + '/%s.epoch%d.test%.5f.state' % (network_name, epoch, epoch_loss)

        if (epoch % save_every == 0):
            print("==> saving ... %s" % state_name)
            dmn.save_params(state_name, epoch)

        print("epoch %d took %.3fs" % (epoch, float(time.time()) - start_time))

    return dmn, network_name

def test_mem_network(dmn, network_name, batch_size=10, log_every=4, **kwargs):

    print("==> testing"   )
    file = open('last_tested_model.json', 'w+')
    data = kwargs
    data["id"] = network_name
    data["name"] = network_name
    data["description"] = ""
    data["vocab"] = list(dmn.vocab.keys())
    json.dump(data, file, indent=2)
    avg_loss, skipped, perform_results, y_pred = do_epoch(dmn, 'test', 0, batch_size, log_every)

    return y_pred, perform_results


def do_epoch(dmn, mode, epoch, batch_size, log_every, skipped=0):
        '''
            This function runs the epochs for the training of the neural network. It calls the steps in the epoch
        and allows for the tweaking of the parameter values.

        Args:
            mode (str): 'train' or 'test' for whether this is the training or testing step
            epoch (int): the epoch number currently on
            skipped (int): how many steps have been skipped because of no change in gradient
        Returns:
            avg_loss (double): the new calculated average loss
            skipped (int): how many steps skipped since the beginning of running the epoch added to the previous skip value
        '''
        # mode is 'train' or 'test'
        y_true = []
        y_pred = []
        avg_loss = 0.0
        prev_time = time.time()

        batches_per_epoch = dmn.get_batches_per_epoch(mode)
        #if batches_per_epoch==0: batches_per_epoch=10
        print(batches_per_epoch, dmn)

        for i in range(0, batches_per_epoch):
            step_data = dmn.step(i, mode) # Run step using the dynamic memory network object
            prediction = step_data["prediction"]
            answers = step_data["answers"]
            current_loss = step_data["current_loss"]
            current_skip = (step_data["skipped"] if "skipped" in step_data else 0)
            log = step_data["log"]

            skipped += current_skip

            if current_skip == 0:
                avg_loss += current_loss

                for x in answers:
                    y_true.append(x)

                for x in prediction.argmax(axis=1):
                    #some predictions are not 0,1 for the first couple of guesses
                    #TODO figure out why...but until then this catches the issue
                    if x not in [0,1]:
                        x = np.random.randint(0,2)
                    y_pred.append(x)

                # TODO: save the state sometimes
                if (i % log_every == 0):
                    cur_time = time.time()
                    print ("  %sing: %d.%d / %d \t loss: %.3f \t avg_loss: %.3f \t skipped: %d \t %s \t time: %.2fs" %
                        (mode, epoch, i * batch_size, batches_per_epoch * batch_size,
                         current_loss, avg_loss / (i + 1), skipped, log, cur_time - prev_time))
                    prev_time = cur_time

            if np.isnan(current_loss):
                print("==> current loss IS NaN. This should never happen :) " )
                exit()

        avg_loss /= batches_per_epoch
        print("\n  %s loss = %.5f" % (mode, avg_loss))
        #print("confusion matrix:")
        #print(metrics.confusion_matrix(y_true, y_pred))

        perform_results = performance_metrics.get_perform_metrics(y_true, y_pred)
        print(perform_results)

        accuracy = sum([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
        print("accuracy: %.2f percent" % (accuracy * 100.0 / batches_per_epoch / batch_size))

        return avg_loss, skipped, perform_results, y_pred