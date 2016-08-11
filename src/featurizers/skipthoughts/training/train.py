"""
Main trainer function
"""
import theano
import theano.tensor as tensor

import pickle as pkl
import numpy
import copy

import os
import warnings
import sys
import time

from training import homogeneous_data

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .utils import *
from .layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer
from .optim import adam
from .model import init_params, build_model
from .vocab import load_dictionary

import logging
time_format = '%Y%m%d %H:%M:%S%Z'
log_format = '%(asctime)s : %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format, datefmt=time_format)

# main trainer
def trainer(X, 
            dim_word=620, # word vector dimensionality
            dim=2400, # the number of GRU units
            encoder='gru',
            decoder='gru',
            max_epochs=5,
            dispFreq=1,
            decay_c=0.,
            grad_clip=5.,
            n_words=20000,
            maxlen_w=30,
            optimizer='adam',
            batch_size = 64,
            saveto='/u/rkiros/research/semhash/models/toy.npz',
            dictionary='/ais/gobi3/u/rkiros/bookgen/book_dictionary_large.pkl',
            saveFreq=1000,
            reload_=False):

    # Model options
    model_options = {}
    model_options['dim_word'] = dim_word
    model_options['dim'] = dim
    model_options['encoder'] = encoder
    model_options['decoder'] = decoder 
    model_options['max_epochs'] = max_epochs
    model_options['dispFreq'] = dispFreq
    model_options['decay_c'] = decay_c
    model_options['grad_clip'] = grad_clip
    model_options['n_words'] = n_words
    model_options['maxlen_w'] = maxlen_w
    model_options['optimizer'] = optimizer
    model_options['batch_size'] = batch_size
    model_options['saveto'] = saveto
    model_options['dictionary'] = dictionary
    model_options['saveFreq'] = saveFreq
    model_options['reload_'] = reload_

    logging.info(model_options)

    # reload options
    if reload_ and os.path.exists(saveto):
        logging.info(('reloading...{}'.format(saveto)))
        with open('%s.pkl' % saveto, 'rb') as f:
            models_options = pkl.load(f)

    # load dictionary
    logging.info('Loading dictionary...')
    worddict = load_dictionary(dictionary)

    # Inverse dictionary
    word_idict = dict()
    for kk, vv in list(worddict.items()):
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    logging.info('Building model')
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    tparams = init_tparams(params)

    trng, x, x_mask, y, y_mask, z, z_mask, \
          opt_ret, \
          cost = \
          build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask, z, z_mask]

    # before any regularizer
    logging.info('Building f_log_probs...')
    f_log_probs = theano.function(inps, cost, profile=False)
    logging.info('Done')

    # weight decay, if applicable
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in list(tparams.items()):
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    logging.info('Building f_cost...')
    f_cost = theano.function(inps, cost, profile=False)
    logging.info('Done')

    logging.info('Done')
    logging.info('Building f_grad...')
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    f_grad_norm = theano.function(inps, [(g**2).sum() for g in grads], profile=False)
    f_weight_norm = theano.function([], [(t**2).sum() for k,t in list(tparams.items())], profile=False)

    if grad_clip > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (grad_clip**2),
                                           g / tensor.sqrt(g2) * grad_clip,
                                           g))
        grads = new_grads

    lr = tensor.scalar(name='lr')
    logging.info('Building optimizers...')
    # (compute gradients), (updates parameters)
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)

    logging.info('Optimization')

    # Each sentence in the minibatch have same length (for encoder)
    trainX = homogeneous_data.grouper(X)
    train_iter = homogeneous_data.HomogeneousData(trainX, batch_size=batch_size, maxlen=maxlen_w)

    uidx = 0
    lrate = 0.01
    for eidx in range(max_epochs):
        n_samples = 0

        logging.info(('Epoch {}'.format(eidx)))

        for x, y, z in train_iter:
            n_samples += len(x)
            uidx += 1

            x, x_mask, y, y_mask, z, z_mask = homogeneous_data.prepare_data(x, y, z, worddict, maxlen=maxlen_w, n_words=n_words)

            if x is None:
                logging.info(('Minibatch with zero sample under length {}'.format(maxlen_w)))
                uidx -= 1
                continue

            ud_start = time.time()
            cost = f_grad_shared(x, x_mask, y, y_mask, z, z_mask)
            f_update(lrate)
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                logging.info('NaN detected')
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
               logging.info(('Epoch {} Update {} Cost {} UD {}'.format(eidx,
                                                               uidx,
                                                               cost,
                                                               ud)))

            if numpy.mod(uidx, saveFreq) == 0:
                logging.info('Saving...')

                params = unzip(tparams)
                numpy.savez(saveto, history_errs=[], **params)
                pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                logging.info('Done')

        logging.info(('Seen {} samples'.format(n_samples)))

if __name__ == '__main__':
    pass


