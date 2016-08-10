from src.utils.normalize import normalize_and_remove_stop_words, xml_normalize
from src.featurizers.skipthoughts import skipthoughts
from src.featurizers import tensorflow_cnn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from src.utils import tokenize
import errno
import os
import sys
import gzip
import shutil
import logging

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

def build_lda(trainingdata, vocabdict, topics=40, **kwargs):
    '''
    Fits a LDA topic model based on the corpus vocabulary.

    Args:
        trainingdata (list): A list containing the corpus as parsed JSON text
        vocabdict (dict): A dictionary containing the vocabulary to be used in the LDA model
        topics (int): the number of topics to be used in the LDA model

    Returns:
        LatentDirichletAllocation: A LDA model fit to the training data and corpus vocabulary
    '''

    vectorizer = CountVectorizer(analyzer = "word", vocabulary = vocabdict)
    trainingdocs = []

    for entry in trainingdata: trainingdocs.append(entry['body_text'])
    trainingvectors = vectorizer.transform(trainingdocs)

    lda_model = LatentDirichletAllocation(n_topics=topics, random_state=0)
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
    if pretrained is True:
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
            w2v_model = gensim.models.Word2Vec.load_word2vec_format(os.path.join(path_to_models, "GoogleNews-vectors-negative300.bin"), binary=True)
        else:
            print("Error: Google's pretrained Word2Vec model GoogleNews-vectors-negative300.bin was not found in %s \nSet 'pretrained=False' or download/unzip GoogleNews-vectors-negative300.bin.gz from https://code.google.com/archive/p/word2vec/ into %s" % (path_to_models,path_to_models), file=sys.stderr)
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

def main(argv):
    '''
    Controls the preprocessing of the corpus, including building vocabulary and model creation.

    Args:
        argv (list): contains a list of the command line features, a dictionary of all tokens in the corpus, an array of parsed JSON documents, a list of the command line parameters

    Returns:
        multiple: dictionary of the corpus vocabulary, skipthoughts encoder_decoder, trained LDA model
    '''

    features, parameters, corpus_dict, trainingdata = argv
    encoder_decoder = None
    vocab= None
    lda_model = None
    tf_session = None
    w2v_model = None

    if 'st' in features: encoder_decoder = skipthoughts.load_model()

    if 'bow' in features or 'lda' in features: vocab = gen_vocab(corpus_dict, **parameters)
    elif 'cnn' in features and 'vocab_type' in features['cnn'] and 'vocab_type' == 'word':
        vocab = gen_vocab(corpus_dict, **features['lda'])
        print("creating a vocab")
        features['lda']['vocab'] = vocab

    if 'lda' in features: lda_model = build_lda(trainingdata, vocab, **features['lda'])

    if 'cnn' in features: tf_session = tensorflow_cnn.tensorflow_cnn(trainingdata, **features['cnn'])

    if 'w2v' in features: w2v_model = build_w2v(trainingdata, **features['w2v'])

    return vocab, encoder_decoder, lda_model, tf_session, w2v_model
