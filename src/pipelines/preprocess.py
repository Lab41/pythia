import os
import sys
import gzip
import shutil
import logging
import errno

import numpy as np
import gensim

from src.utils.normalize import normalize_and_remove_stop_words, xml_normalize
#from src.featurizers.skipthoughts import skipthoughts
#from src.featurizers import tensorflow_cnn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from src.utils import tokenize

def gen_vocab(corpus_dict, vocab=1000, stem=False, **kwargs):
    '''
    Generates a dictionary of words to be used as the vocabulary in features that utilize bag of words.

    Args:
        corpus_dict (OrderedDict): An ordered list of the most frequently occurring tokens in the corpus
        vocab_size (int): the number of words to be used in the vocabulary

    Returns:
        dict: a dictionary of size vocab_size that contains the most frequent normalized and non-stop words in the corpus
    '''
    index = 0
    vocabdict = dict()
    for word in corpus_dict:
        if len(vocabdict) < vocab:
            cleantext = normalize_and_remove_stop_words(word, stem)
            if cleantext != '':
                if not cleantext in vocabdict:
                    vocabdict[cleantext] = index
                    index+=1
        else: break
    return vocabdict

def gen_full_vocab(corpus_dict, full_vocab_type='word', full_vocab_size=1000, full_vocab_stem=False, full_char_vocab="", token_include = {'.',',','!','?'}, **kwargs):
    '''
    Generates a dictionary of words to be used as the vocabulary in features that utilize bag of words.
    This vocab contains stop words and punctuation

    Args:
        corpus_dict (OrderedDict): An ordered list of the most frequently occurring tokens in the corpus
        vocab_size (int): the number of words to be used in the vocabulary

    Returns:
        dict: a dictionary of size vocab_size that contains the most frequent normalized and non-stop words in the corpus
    '''

    vocabdict = dict()
    if full_vocab_type=='character':
        index=0
        for c in full_char_vocab:
            vocabdict[c] = index
            index+= 1

    else:
        index = 0
        vocabdict = dict()
        for word in corpus_dict:
            if len(vocabdict) < full_vocab_size:
                cleantext = xml_normalize(word, full_vocab_stem)
                if cleantext != '':
                    if not cleantext in vocabdict:
                        vocabdict[cleantext] = index
                        index+=1
            else: break

    #For each of these we need to ensure that the punctuation or list of tokens we desire is in the dictionary
    for t in token_include:
        if t not in vocabdict.keys():
            vocabdict[t] = index
            index+=1

    return vocabdict

def build_lda(trainingdata, vocabdict, topics=40, random_state=0, **kwargs):
    '''
    Fits a LDA topic model based on the corpus vocabulary.

    Args:
        trainingdata (list): A list containing the corpus as parsed JSON text
        vocabdict (dict): A dictionary containing the vocabulary to be used in the LDA model
        topics (int): the number of topics to be used in the LDA model
        random_state (int or np.random.RandomState): seed value or random number generator state

    Returns:
        LatentDirichletAllocation: A LDA model fit to the training data and corpus vocabulary
    '''

    vectorizer = CountVectorizer(analyzer = "word", vocabulary = vocabdict)
    trainingdocs = []

    for entry in trainingdata: trainingdocs.append(entry['body_text'])
    trainingvectors = vectorizer.transform(trainingdocs)

    lda_model = LatentDirichletAllocation(n_topics=topics, random_state=random_state)
    lda_model.fit(trainingvectors)
    return lda_model

def build_w2v(trainingdata, min_count=5, window=5, size=100, workers=3, pretrained=False, **kwargs):
    '''
    Fits a Word2Vec topic model based on the training corpus sentences.

    Args:
        trainingdata (list): A list containing the training corpus as parsed JSON text
        min_count (int): ignore all words with total frequency lower than this number
        window (int): maximum distance between the current and predicted word within a sentence
        size (int): dimensionality of the feature vectors
        workers (int): use this many worker threads to train the model (faster training with multicore machines)

    Returns:
        Word2Vec: A pretrained Word2Vec model from Google or a Word2Vec model fit to the training data sentences
    '''

    # Suppress gensim's INFO messages
    logging.getLogger("gensim").setLevel(logging.WARNING)

    # Use Google's pretrained Word2Vec model
    if pretrained:
        # Look at environment variable 'PYTHIA_MODELS_PATH' for user-defined model location
        # If environment variable is not defined, use current working directory
        if os.environ.get('PYTHIA_MODELS_PATH') is not None:
            path_to_models = os.environ.get('PYTHIA_MODELS_PATH')
        else:
            path_to_models = os.path.join(os.getcwd(), 'models')
        # Make the directory for the models unless it already exists
        try:
            os.makedirs(path_to_models)
        except OSError as exception:
            if exception.errno != errno.EEXIST: raise
        # Look for Google's trained Word2Vec model as a binary or zipped file; Return error and quit if not found
        if os.path.isfile(os.path.join(path_to_models, "GoogleNews-vectors-negative300.bin")):
            w2v_model = gensim.models.Word2Vec.load_word2vec_format(os.path.join(path_to_models, "GoogleNews-vectors-negative300.bin"), binary=True)
        elif os.path.isfile(os.path.join(path_to_models, "GoogleNews-vectors-negative300.bin.gz")):
            with gzip.open(os.path.join(path_to_models, "GoogleNews-vectors-negative300.bin.gz"), 'rb') as f_in:
                with open(os.path.join(path_to_models, "GoogleNews-vectors-negative300.bin"), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            w2v_model = gensim.models.Word2Vec.load_word2vec_format(
                os.path.join(path_to_models, "GoogleNews-vectors-negative300.bin"), binary=True)
        else:
            print("""Error: Google's pretrained Word2Vec model GoogleNews-vectors-negative300.bin was not found in %s
Set 'pretrained=False' or download/unzip GoogleNews-vectors-negative300.bin.gz 
from https://code.google.com/archive/p/word2vec/ into %s""" % (path_to_models,path_to_models), file=sys.stderr)
            quit()

    # Train a Word2Vec model with the corpus
    else:
        sentencearray = []
        for entry in trainingdata:
            sentences = tokenize.punkt_sentences(xml_normalize(entry['body_text']))
            for sentence in sentences:
                words = tokenize.word_punct_tokens(sentence)
                sentencearray.append(words)

        w2v_model = gensim.models.Word2Vec(sentencearray, min_count=min_count, window=window, size=size, workers=workers)

    return w2v_model

def main(features, parameters, corpus_dict, trainingdata):
    '''
    Controls the preprocessing of the corpus, including building vocabulary and model creation.

    Args:
        argv (list): contains a list of the command line features, a dictionary of all 
        tokens in the corpus, an array of parsed JSON documents, a list of the command line parameters

    Returns:
        multiple: dictionary of the corpus vocabulary, skipthoughts encoder_decoder, trained LDA model
    '''

    encoder_decoder = None
    vocab= None
    lda_model = None
    tf_session = None
    w2v_model = None
    full_vocab = None

    if 'st' in features:
        from src.featurizers.skipthoughts import skipthoughts
        encoder_decoder = skipthoughts.load_model()

    if 'bow' in features or 'lda' in features:
        vocab = gen_vocab(corpus_dict, **parameters)

    if 'cnn' in features:
        from src.featurizers import tensorflow_cnn
        full_vocab = gen_full_vocab(corpus_dict, **parameters)
        features['cnn']['vocab'] = full_vocab
        tf_session = tensorflow_cnn.tensorflow_cnn(trainingdata, **features['cnn'])

    if 'lda' in features: 
        features['lda']['vocab'] = vocab
        lda_model = build_lda(trainingdata, vocab, 
            random_state=parameters['seed'], **features['lda'])

    if 'w2v' in features: w2v_model = build_w2v(trainingdata, **features['w2v'])

    if 'wordonehot' in features: full_vocab = gen_full_vocab(corpus_dict, **parameters)

    #get the appropriate model(s) when running the memory network code
    if 'mem_net' in features:
        if features['mem_net'].get('embed_mode', False):
            embed_mode = features['mem_net']['embed_mode']
        else: embed_mode = 'word2vec'
        if embed_mode=='skip_thought' and not encoder_decoder:
            from src.featurizers.skipthoughts import skipthoughts
            encoder_decoder = skipthoughts.load_model()
        if embed_mode=="onehot" and not full_vocab:
            full_vocab = gen_full_vocab(corpus_dict, **parameters)
        if embed_mode=='word2vec' and not w2v_model:
            w2v_model = build_w2v(trainingdata, **features['mem_net'])

    return vocab, full_vocab, encoder_decoder, lda_model, tf_session, w2v_model
