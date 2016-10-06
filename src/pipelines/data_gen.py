import os
import copy
import logging
import math
import h5py
import numpy as np
from memory_profiler import profile
from scipy import spatial

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder
#from src.featurizers.skipthoughts import skipthoughts as sk
from src.utils import normalize, tokenize, sampling

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

def gen_feature(new_vectors, request_parameters, feature_vector):
    """Take newly generated feature vectors, look up which
    transformations have been requested for them, and append the
    transformed feature vectors to the existing feature collection.

    Args:
        new_vectors (list, np.NDArray): feature vectors to be transformed
        request_parameters (dict): should feature vectors be concatenated,
            subtracted, multiplied, or other forms of comparison made?
        feature_vector (list?): existing list of feature vectors
     """
    if request_parameters.get('append', False):
        feature_vector.append(np.concatenate(new_vectors, axis=0))
    if request_parameters.get('difference', False):
        feature_vector.append(np.subtract(new_vectors[0], new_vectors[1]))
    if request_parameters.get('product', False):
        feature_vector.append(np.multiply(new_vectors[0], new_vectors[1]))
    if request_parameters.get('cos', False):
        similarity = 1 - spatial.distance.cosine(new_vectors[0], new_vectors[1])
        # Set similarity to zero when zero vector(s) result in cosine distance of NaN/Inf
        if np.isnan(similarity) or np.isinf(similarity):
            similarity = 0                
        feature_vector.append(np.array([similarity]))
    return feature_vector

def bow(doc, corpus, corpus_array, vocab, bow, feature):
    if bow.get('binary', False):
        binary_bow= bow['binary']
    else: binary_bow = False
    vectors = bag_of_words_vectors(doc, corpus, vocab, binary_bow)
    feature = gen_feature(vectors, bow, feature)
    if 'tfidf' in bow:
        feature = tfidf_sum(doc, corpus_array, vocab, feature)
    return feature

def bag_of_words_vectors(doc, corpus, vocab, binary):
    '''
    Creates bag of words vectors for doc and corpus for a given vocabulary.

    Args:
        doc (str): the text (normalized and without stop words) of the document
        corpus (str): the text (normalized and without stop words) of the corpus for that cluster
        vocab (dict): the vocabulary of the data set

    Returns:
        array: contains the bag of words vectors
    '''

    # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
    #http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vectorizer = CountVectorizer(analyzer = "word",  \
                                 vocabulary = vocab, binary=binary)

    # Combine Bag of Words dicts in vector format, calculate cosine similarity of resulting vectors
    bagwordsVectors = (vectorizer.transform([doc, corpus])).toarray()
    return bagwordsVectors

def tfidf_sum(doc, corpus_array, vocab, feature):
    '''
    Calculates L1 normalized TFIDF summation as Novelty Score for new document against corpus.

    Credit to http://cgi.di.uoa.gr/~antoulas/pubs/ntoulas-novelty-wise.pdf

    Args:
        doc (str): the text (normalized and without stop words) of the document
        corpus (str): the text (normalized and without stop words) of the corpus for that cluster (including the current doc)

    Returns:
        float: the normalized TFIDF summation
    '''
    doc_array = tokenize.word_punct_tokens(doc)
    doc_length = len(doc_array)
    if doc_length != 0:
        vectorizer = TfidfVectorizer(norm=None, vocabulary = vocab)
        tfidf = vectorizer.fit_transform(corpus_array)
        vector_values = tfidf.toarray()
        tfidf_score = np.sum(vector_values[-1])/doc_length
        feature.append(np.array([tfidf_score]))
    else:
        feature.append(np.zeros(1))
    return feature


def st(doc, sentences, encoder_decoder, st, feature):
    vectors = skipthoughts_vectors(doc, sentences, encoder_decoder)
    feature = gen_feature(vectors, st, feature)
    return feature

def skipthoughts_vectors(doc, sentences, encoder_decoder):
    '''
    Creates skipthoughts vector for doc and corpus for a given encoder_decoder

    The encode function produces an array of skipthought vectors with as many rows as there were sentences and 4800 dimensions.
    See the combine-skip section of the skipthoughts paper for a detailed explanation of the array.

    Args:
        doc (str): the text of the document (before any normalization)
        corpus (list): the first and last sentences of each document in the corpus
        encoder_decoder (???): the skipthoughts encoder/decoder

    Returns:
        array: the concatenation of the corpus skipthoughts vector (the average of each indivdual skipthoughts vector) and the document skipthoughts vector (the average of the first and last sentence's skipthoughts vector)
    '''
    from src.featurizers.skipthoughts import skipthoughts as sk
    corpus_vectors = sk.encode(encoder_decoder, sentences)
    corpus_vector = np.mean(corpus_vectors, axis = 0)
    doc_vector = np.mean(sk.encode(encoder_decoder, get_first_and_last_sentence(doc)), axis=0)
    skipthoughts = [doc_vector, corpus_vector]
    return skipthoughts

def get_first_and_last_sentence(doc):
    '''
    Finds the first and last sentance of a document and normalizes them.

    Args:
        doc (str): the text of the document (before any preprocessing)

    Returns:
        array: the first and last sentance after normalizing
    '''
    sentences = tokenize.punkt_sentences(doc)
    first = normalize.xml_normalize(sentences[0])
    last = normalize.xml_normalize(sentences[-1])

    # Protect against scenario where last sentence is mistakenly returned by parser as empty list
    if len(last)==0:
        i = -2
        while len(last)==0:
            last = normalize.xml_normalize(sentences[i])
            i-=1

    first_and_last = [first, last]
    return first_and_last

def lda(doc, corpus, vocab, lda_topics, lda, feature):
    doclda = run_lda(lda_topics, doc, vocab)
    corpuslda = run_lda(lda_topics, corpus, vocab)
    vectors = [doclda, corpuslda]
    feature = gen_feature(vectors, lda, feature)
    return feature

def run_lda(lda_topics, doc, vocab):
    '''
    Calculates a vector of topic probabilities for a single document based on a trained LDA model.

    Args:
        lda_topics (LatentDirichletAllocation): A LDA model previously fit to vocabulary of training data
        doc (str): the text (normalized and without stop words) of the document
        vocab (dict): the vocabulary of the data set

    Returns:
        array: a vector of topic probabilities based on a trained LDA model
    '''

    vectorizer = CountVectorizer(analyzer = "word", vocabulary = vocab)
    docvector = vectorizer.transform([doc])
    return lda_topics.transform(docvector)[0]

def w2v(doc, background_docs, w2v_model, w2v, feature):
    '''
     Calculates Word2Vec features for a document and corpus

     Args:
         doc (str): the text of the document (normalized but with stop words and punctuation)
         background_docs (list): background documents (normalized but with stop words and punctuation)
         w2v_model (gensim.Word2Vec): Trained Word2Vec model
          w2v (dict): Dictionary of Word2Vec parameters as set in master_pipeline. The dictionary
           will include keys for the model building parameters min_count, window, size, workers and pretrained.
           The dict may also have optional boolean keys for the feature operations append, difference, product and cos.
         feature (list): List of features extracted from text

     Returns:
         feature: List of features extracted from text
     '''

    if w2v.get('avg', False):
        docw2v = run_w2v(w2v_model, doc, w2v)
        background_vectors = list()
        for item in background_docs:
            background_vectors.append(run_w2v(w2v_model, item, w2v))
        backgroundw2v = np.mean(background_vectors, axis=0)
        vectors = [docw2v, backgroundw2v]
        feature = gen_feature(vectors, w2v, feature)

    vectormath = []
    if w2v.get('max', False): vectormath.append('max')
    if w2v.get('min', False): vectormath.append('min')
    if w2v.get('abs', False): vectormath.append('abs')
    for operation in vectormath:
        docw2v = run_w2v_elemwise(w2v_model, doc, w2v, operation)
        background_vectors = list()
        for entry in background_docs:
            background_vectors.append(run_w2v_elemwise(w2v_model, entry, w2v, operation))
        if operation == 'min':
            backgroundw2v = np.amin(background_vectors, axis=0)
        elif operation == 'max':
            backgroundw2v = np.amax(background_vectors, axis=0)
        elif operation == 'abs':
            backgroundw2v = np.amax(np.fabs(background_vectors), axis=0)
        vectors = [docw2v,backgroundw2v]
        feature = gen_feature(vectors, w2v, feature)
    return feature

def run_w2v_elemwise(w2v_model, doc, w2v, operation):
    '''
      Calculates Word2Vec vectors for a document using the first and last sentences of the document
      Examines vector elements and retains maximum, minimum or absolute value for each vector element

      Args:
          w2v_model (gensim.Word2Vec): Trained Word2Vec model
          doc (str): the text of the document
          w2v (dict): Dictionary of Word2Vec parameters as set in master_pipeline. The dictionary
           will include keys for the model building parameters min_count, window, size, workers and pretrained.
           The dict may also have optional boolean keys for the feature operations append, difference, product and cos.
          operation (str): element wise operation of max, min or abs
      Returns:
          documentvector (list): Word2Vec vectors with min/max/abs element values for a sentence, which are then
          concatenated across sentences
      '''
    # Get first and last sentences of document, break down sentences into words and remove stop words

    sentences = get_first_and_last_sentence(doc)
    sentencevectorarray = []

    # Look up word vectors in trained Word2Vec model and build array of word vectors and sentence vectors
    for phrase in sentences:

        # Set up comparison vector based on requested operation
        if operation == 'max':
            vectorlist = np.full(w2v['size'], -np.inf)
        elif operation == 'min':
            vectorlist = np.full(w2v['size'], np.inf)
        elif operation == 'abs':
            vectorlist = np.zeros(w2v['size'])

        # Determine word vector and evaluate elements against comparison vector
        for word in phrase:
            try:
                wordvector = w2v_model[word]
            except KeyError:
                continue
            if operation == 'max':
                vectorlist = np.where(wordvector > vectorlist, wordvector, vectorlist)
            elif operation == 'min':
                vectorlist = np.where(wordvector < vectorlist, wordvector, vectorlist)
            elif operation == 'abs':
                vectorlist = np.where(abs(wordvector) > vectorlist, abs(wordvector), vectorlist)

        # Remove any infinity values from special cases (ex: 1 word sentence and word not in word2vec model)
        vectorlist = np.where(np.isinf(vectorlist), 0, vectorlist)

        sentencevectorarray.append(vectorlist)

    # Only concatenate if both sentences were added to sentence vector array, otherwise append array of zeroes
    if len(sentencevectorarray) == 2:
        documentvector = np.concatenate(sentencevectorarray)
    elif len(sentencevectorarray) == 1:
        documentvector = np.concatenate((sentencevectorarray[0], np.zeros(w2v['size'])))
    else:
        documentvector = np.zeros(w2v['size']*2)
    return documentvector

def run_w2v(w2v_model, doc, w2v):
    '''
      Calculates Word2Vec vectors for a document using the first and last sentences of the document

      Args:
          w2v_model (gensim.Word2Vec): Trained Word2Vec model
          doc (str): the text of the document
          w2v (dict): Dictionary of Word2Vec parameters as set in master_pipeline. The dictionary
           will include keys for the model building parameters min_count, window, size, workers and pretrained.
           The dict may also have optional boolean keys for the feature operations append, difference, product and cos.

      Returns:
          documentvector (list): List of Word2Vec vectors averaged across words and concatenated across sentences
      '''

    # Get first and last sentences of document, break down sentences into words and remove stop words
    sentences = get_first_and_last_sentence(doc)
    wordvectorarray = []
    sentencevectorarray = []

    # Look up word vectors in trained Word2Vec model and build array of word vectors and sentence vectors
    for phrase in sentences:
        for word in phrase:
            try:
                wordvector = w2v_model[word]
            except KeyError:
                continue
            wordvectorarray.append(wordvector)

        # Only calculate mean and append to sentence vector array if one or more word vectors were found
        if len(wordvectorarray) > 0:
            sentencevectorarray.append(np.mean(wordvectorarray, axis=0))

    # Only concatenate if both sentences were added to sentence vector array, otherwise append array of zeroes
    if len(sentencevectorarray) == 2:
        documentvector =  np.concatenate(sentencevectorarray)
    elif len(sentencevectorarray) == 1:
        documentvector =  np.concatenate((sentencevectorarray[0], np.zeros(w2v['size'])))
    else:
        documentvector = np.zeros(w2v['size']*2)
    return documentvector

def run_w2v_matrix(w2v_model, doc, w2v_params, mask_mode):

    #determine if the first and last sentences will be taken or all sentences
    if w2v_params.get('mem_w2v_mode', False):
        w2v_mode = w2v_params['mem_w2v_mode']
    else: w2v_mode = 'all'

    if w2v_mode == 'all':
        sentences = tokenize.punkt_sentences(doc)
    else:
        sentences = get_first_and_last_sentence(doc)

    normalizedsentences = []

    sentence_mask = []
    for sentence in sentences:
        words = tokenize.word_punct_tokens(sentence)
        if len(sentence_mask)>0: 
            prev_mask = sentence_mask[-1]
        else: 
            prev_mask = -1
        sentence_mask.append(prev_mask + len(words))
        normalizedsentences.append(words)

    wordvectorarray = []

    # Look up word vectors in trained Word2Vec model and build array of word vectors and sentence vectors
    for phrase in normalizedsentences:
        for word in phrase:
            wordvector = None
            try:
                wordvector_ = w2v_model[word]
                wordvector = [float(w) for w in wordvector_]
            except:
                wordvector = w2v_model.seeded_vector(np.random.rand())
            if wordvector is not None:
                wordvectorarray.append(wordvector)

    if mask_mode=='sentence': 
        mask = sentence_mask
    else:
        mask = np.array([index for index, w in enumerate(wordvectorarray)], dtype=np.int32)

    if len(wordvectorarray)-1!=mask[-1]:
        print(mask)
        print(np.array(wordvectorarray).shape)
        raise

    return np.vstack(wordvectorarray), mask

def run_cnn(doc, corpus, tf_session):
    doc_cnn, corpus_cnn = tf_session.transform_doc(doc, corpus)

    return [doc_cnn, corpus_cnn]


    return feature

def run_onehot(doc, vocab, min_length=None, max_length=None, already_encoded=False):
    """ One-hot encode array of tokens, given a vocabulary mapping
    them to 0-to-n integer space

    Args:
        doc (list): list of tokens; should correspond to the keys in vocab (so,
            typically str)
        vocab (dict): map of vocab items to integers (zero based)
        min_length: if not None, enforce a minimum document length by zero-padding
            the right edge of the result
        max_length: if not None, truncate documents to max_length
        already_encoded (bool): if True, skip encoding step and treat
            doc as onehot-encoded NDArray

    Returns:
        NDArray (vocab size, doc length), with 1 indicating presence of vocab item
            at that position. Out-of-vocab entries do not appear in the result.
    """
    if not already_encoded:
        doc_indices = encode_doc(doc, vocab, oov_strategy='skip')
        vocab_size = len(vocab)
        doc_onehot = onehot_encode(doc_indices, vocab_size)
    else:
        vocab_size = len(vocab)
        doc_onehot = doc
    doc_length = doc_onehot.shape[1]
    # Zero-padding if doc is too short
    if min_length is not None and doc_length < min_length:
        padding_size = (vocab_size, min_length - doc_length)
        doc_onehot = np.concatenate((doc_onehot, np.zeros(padding_size)), axis=1)
        doc_length = doc_onehot.shape[1]
    # Truncate if document is too long
    if max_length is not None and doc_length > max_length:
        doc_onehot = doc_onehot[:, :max_length]
        doc_length = doc_onehot.shape[1]


    return doc_onehot

def onehot_encode(doc, size):
    ''' Encode list of indices in one-hot format, producing a sparse
    matrix of binary codes

    Args:
        doc (list): indices to 'flip on' in one-hot encoding
        size (int): size of one-hot vectors to create
    '''
    doc_length = len(doc)
    doc_onehot = np.zeros((size, doc_length), dtype=np.float32)
    for token_idx, token in enumerate(doc):
        doc_onehot[token, token_idx] = 1
    return doc_onehot

def encode_doc(doc, vocab, oov_strategy='skip'):   
    """
    Integer-encode doc according to vocab. Options for 
    how to treat out-of-vocabulary tokens

    Args:
        doc (list): list of tokens to encode
        vocab (dict): mapping of tokens to integer codes
        oov_strategy (str or int): if 'skip', leave out-of-vocab tokens
            out of result. If 'none', replace oov tokens with None. If 
            any integer, replace oov tokens with that integer.
    Returns:
        list of integers (and possibly None)
    """

    if oov_strategy == 'skip':
        doc = strip_to_vocab(doc, vocab)
        oov_code = None
    elif type(oov_strategy) is int:
        oov_code = oov_strategy
    elif oov_strategy is None:
        oov_code = None
        
    encoded_doc = [ vocab.get(tkn, oov_code) for tkn in doc ]
    return encoded_doc

def get_mask(doc_idxs, vocab, dividers = ['.', '!', '?'], add_final_posn=True, max_length=None):
    """ Return the indices from a integer-encoded document
    representing the non-contiguous instances of divider characters

    Args:
        doc_idxs (list): document to mask, encoded as int according to the mapping in vocab
        vocab (dict): map of token (str) to id (int)
        dividers (list of str): which characters to divide on?
        add_final_posn (bool): Add an index for the last posn in doc_idxs, even if not a divider

    Returns:
        list of list indices where dividers occur
    """

    doc_length = len(doc_idxs)
    last_tkn_was_mask = False
    sentence_mask = []
    divider_idx = set(vocab[divider] for divider in dividers)
    for idx, tkn in enumerate(doc_idxs):
        if tkn in divider_idx and not last_tkn_was_mask:
            last_tkn_was_mask = True
            sentence_mask.append(idx)
        else:
            last_tkn_was_mask = False
    #check to ensure there are no mask values greater than the maximum value
    if max_length and doc_length-1>max_length - 1:
        max_mask = max_length - 1
    else:
        max_mask = doc_length-1
    sentence_mask = [a for a in sentence_mask if a<max_mask]
    if add_final_posn:
        # make sure to add in the last index if it is not already there
        if len(sentence_mask)==0 or sentence_mask[-1] != max_mask:
            sentence_mask.append(max_mask)
    return sentence_mask

def remove_by_position(doc, idxs):
    """Remove elements from document by position
    and return the result alongside with the list index 
    of the last element before each removed element
    
    Args:
        doc (list): list of tokens
        idxs (list): indices in doc to remove
    
    Returns:
        list, list: doc, with elements at idxs removed, and 
          and adjusted list of indices into the modified doc"""
    
    idxs = sorted(idxs)
    # new indices are offset by 1 + however many indices come before them 
    mask_idxs = [ i - (1 + offset) for offset, i in enumerate(idxs) ]

    masked_doc = []
    for idx, last_idx in zip(idxs, [-1] + idxs[:-1]):
        masked_doc.extend(doc[last_idx + 1:idx])
    return masked_doc, mask_idxs 

def strip_to_vocab(doc, vocab):
    """ Remove from doc any tokens not in vocab.

    Args:
        doc (list): list of tokens
        vocab (dict): keys overlap with tokens in doc

    Returns:
        list
    """
    return [ tkn for tkn in doc if tkn in vocab ]

def wordonehot(doc, corpus, vocab, transformations, feature, min_length=None, max_length=None):
    # Normalize and tokenize the text before sending it into the one-hot encoder
    norm_doc = tokenize.word_punct_tokens(normalize.xml_normalize(doc))
    norm_corpus = tokenize.word_punct_tokens(normalize.xml_normalize(corpus))
    doc_onehot = run_onehot(norm_doc, vocab, min_length, max_length)
    corpus_onehot = run_onehot(norm_corpus, vocab, min_length, max_length)
    feature = gen_feature([doc_onehot, corpus_onehot], transformations, feature)
    return feature

def gen_mem_net_observations(raw_doc, raw_corpus, sentences_full, mem_net_params, vocab, full_vocab, w2v_model, encoder_decoder):
    '''
    Generates observations to be fed into the mem_net code

    Args:
        raw_doc (string): the raw document text
        raw_corpus (str): the raw corpus text
        sentences_full (list): list of all sentences in the corpus
        mem_net_params (dict): the specified features to be calculated for mem_net
        vocab (dict): the vocabulary of the data set
        w2v_model: the word2vec model of the data set
        encoder_decoder (???): the encoder/decoder for skipthoughts vectors

    Returns:
        doc_input (array): the corpus data, known in mem_nets as the input
        doc_questions: the document data, known in mem_nets as the question
        doc_masks: the mask for the input data - tells mem_net where the end of each input is
            this can be per word for the end of a sentence
     '''

    # Use the specified mask mode where available
    if mem_net_params.get('mask_mode', False):
        mask_mode = mem_net_params["mask_mode"]
    else: mask_mode = 'sentence'

    if mem_net_params.get('embed_mode', False):
        embed_mode = mem_net_params['embed_mode']
    else: embed_mode = 'word2vec'

    if embed_mode == 'skip_thought':
        from src.featurizers.skipthoughts import skipthoughts as sk
        doc_sentences = tokenize.punkt_sentences(raw_doc)

        # Ensure that the document and corpus are long enough and if not make them be long enough
        if len(sentences_full)==1:
            #print("short corpus")
            sentences_full.extend(sentences_full)
        if len(doc_sentences)==1:
            #print("short doc")
            doc_sentences.extend(doc_sentences)
        corpus_vectors = sk.encode(encoder_decoder, sentences_full)
        doc_vectors = sk.encode(encoder_decoder, doc_sentences)

        # Since each entry is a sentence, we use the index of each entry for the mask
        # We cannot use a word mode in this embedding
        doc_masks = [index for index, w in enumerate(corpus_vectors)]
        doc_questions = doc_vectors
        doc_input = corpus_vectors


    elif embed_mode == 'onehot':
        min_length = None
        max_length = None
        if mem_net_params.get('onehot_min_len', False):
            min_length = mem_net_params['onehot_min_len']
        if mem_net_params.get('onehot_max_len', False):
            max_length = mem_net_params['onehot_max_len']
        onehot_vocab=full_vocab

        # Preprocess and tokenize bkgd documents
        corpus_tokens = tokenize.word_punct_tokens(normalize.xml_normalize(raw_corpus))
        corpus_tokens = strip_to_vocab(corpus_tokens, onehot_vocab)
        corpus_indices = encode_doc(corpus_tokens, onehot_vocab)
        # Get sentence mask indices
        assert {'.',',','!','?'} <= onehot_vocab.keys()  # ensure that you are using a vocabulary w/ punctuation
        sentence_mask = get_mask(corpus_indices, onehot_vocab, max_length=max_length)
        # One-hot encode documents w/ masks, and query document
        corpus_encoded = onehot_encode(corpus_indices, len(onehot_vocab))
        corpus_vectors = run_onehot(corpus_encoded, onehot_vocab, min_length, max_length, already_encoded=True)
        # Tokenize and  one-hot encode query document
        doc_vectors = run_onehot(tokenize.word_punct_tokens(normalize.xml_normalize(raw_doc)), 
                                    onehot_vocab, min_length, max_length)

        doc_questions = doc_vectors.T
        doc_input = corpus_vectors.T

        if mask_mode=='sentence':
            doc_masks = sentence_mask
        else: doc_masks = [index for index, w in enumerate(doc_input)]


    elif embed_mode == 'word2vec':
        corpus_vectors, doc_masks = run_w2v_matrix(w2v_model, raw_corpus, mem_net_params, mask_mode)
        doc_vectors, _ = run_w2v_matrix(w2v_model, raw_doc, mem_net_params, mask_mode)

        if len(corpus_vectors)>0 and len(doc_vectors)>0:
            doc_questions = doc_vectors
            doc_input = corpus_vectors

    return doc_input, doc_questions, doc_masks

def gen_observations(all_clusters, lookup_order, document_data, features, parameters, vocab, full_vocab, encoder_decoder, lda_model, tf_session, w2v_model, hdf5_path=None, dtype=np.float32):
    '''
    Generates observations for each cluster found in JSON file and calculates the specified features.

    Args:
        all_clusters (set): cluster IDs
        lookup_order (dict): document arrival order
        document_data (array): parsed JSON documents
        features (dict): the specified features to be calculated
        parameters (dict): data structure with run parameters
        vocab (dict): the vocabulary of the data set
        full_vocab (dict_: to vocabulary of the data set including stop wrods and punctuation
        encoder_decoder (???): the encoder/decoder for skipthoughts vectors
        lda_model (sklearn.???): trained LDA model
        tf_session: active TensorFlow session
        w2v_model (gensim.word2vec): trained word2vec model

    Returns:
        data(list): contains for each obeservation the features of the document vs corpus which could include:
            tfidf sum, cosine similarity, bag of words vectors, skip thoughts, lda, w2v or, onehot cnn encoding
        labels(list): the labels for each document where a one is novel and zero is duplicate
    '''

    # Prepare to store results of feature assessments
    data = list()
    labels = list()
    # mem_net_features is used when the mem_net algorithm is ran
    # It consist of inputs, labels(answers), input_masks and questions for each entry
    mem_net_features = {}
    inputs = []
    input_masks = []
    questions = []
    # Sentence punctuation delimiters
    punkt = ['.','?','!']

    corpus_unprocessed = list()
    # HDF5-related parameters
    hdf5_save_frequency=parameters['hdf5_save_frequency']
    data_key = 'data'
    labels_key = 'labels'
    # Truncate any existing files at save location, or return early if 
    # using existing files
    if hdf5_path is not None:
        if parameters['hdf5_use_existing'] and os.path.isfile(hdf5_path):
            return hdf5_path, hdf5_path
        open(hdf5_path, 'w').close()

    # Create random state
    random_state = np.random.RandomState(parameters['seed'])

    # Iterate through clusters found in JSON file, generate observations
    # pairing data and label
    for cluster in all_clusters:
        # Determine arrival order in this cluster
        sorted_entries = [x[1] for x in sorted(lookup_order[cluster], key=lambda x: x[0])]
        observations = [document_data[sorted_entries[0]]]
        for index in sorted_entries[1:]:
            next_doc = document_data[index]
            observations.append(next_doc)
            labeled_observation = { 'novelty' : next_doc['novelty'],
                    'data' : copy.copy(observations) }
            corpus_unprocessed.append(labeled_observation)
    
    # Resample if necessary
    # If oversampling +/- replacement, sample up
    # to larger class size for both classes, with replacement
    # If -oversampling, sample down to 
    # smaller class size for both classes with or w/o replacement
    if 'resampling' in parameters:
        resampling_parameters = parameters['resampling']
        if resampling_parameters.get('over', False):
            desired_size = None
            resampling_parameters['replacement'] = True
        else:
            desired_size = -np.Inf
        if resampling_parameters.get('replacement', False):
            replacement = True
        else:
            replacement = False
        logger.debug("Replacement: {}, Desired size: {}".format(replacement, desired_size))
        logger.debug("Size of data: {}, Number of clusters: {}".format(len(corpus_unprocessed), len(all_clusters)))
        corpus = sampling.label_sample(corpus_unprocessed, "novelty", replacement, desired_size, random_state)  
    else:
        corpus = corpus_unprocessed

    # Featurize each observation
    # Some duplication of effort here bc docs will appear multiple times 
    # across observations
    
    clusterids = []
    postids = []
    for case in corpus:
        
        # Create raw and normalized document arrays
        case_docs_raw = [ record['body_text'] for record in case['data'] ]
        case_docs_normalized = [ normalize.xml_normalize(body_text) for body_text in case_docs_raw ]
        case_docs_no_stop_words = [ normalize.normalize_and_remove_stop_words(body_text) for body_text in case_docs_raw ]
        #create ids for individual data points
        postid = [record['post_id'] for record in case['data'] ][-1]
        postids.append(postid)
        clusterid = [ record['cluster_id'] for record in case['data'] ][0]
        clusterids.append(clusterid)
        # Pull out query documents
        doc_raw = case_docs_raw[-1]
        doc_normalized = case_docs_normalized[-1]
        doc_no_stop_words = case_docs_no_stop_words[-1]
        # Create lists of background documents
        bkgd_docs_raw = case_docs_raw[:-1]
        bkgd_docs_normalized = case_docs_normalized[:-1]
        bkgd_docs_no_stop_words = case_docs_no_stop_words[:-1]
        bkgd_text_raw = '\n'.join(bkgd_docs_raw)
        bkgd_text_normalized = '\n'.join(bkgd_docs_normalized) 
        bkgd_text_no_stop_words = '\n'.join(bkgd_docs_no_stop_words)
        feature_vectors = list()

        if 'mem_net' in features:
            # Get all sentences for the memory network algorithm
            bkgd_sentences_full = tokenize.punkt_sentences(bkgd_text_raw)
            doc_input, doc_questions, doc_masks = gen_mem_net_observations(doc_raw, bkgd_text_raw, bkgd_sentences_full, features['mem_net'], vocab, full_vocab, w2v_model, encoder_decoder)

            # Now add all of the input docs to the primary list
            inputs.append(doc_input)
            questions.append(doc_questions)
            input_masks.append(doc_masks)

        else:

            if 'bow' in features:
                feature_vectors = bow(doc_no_stop_words, bkgd_text_no_stop_words,
                    bkgd_docs_no_stop_words, vocab, features['bow'], feature_vectors)
            if 'st' in features:
                sentences = []
                for doc in bkgd_docs_raw:
                    for item in get_first_and_last_sentence(doc):
                        sentences.append(item)
                feature_vectors = st(doc_raw, sentences, encoder_decoder, features['st'], feature_vectors)

            if 'lda' in features:
                feature_vectors = lda(doc_no_stop_words, bkgd_text_no_stop_words, vocab, lda_model, features['lda'], feature_vectors)

            if 'w2v' in features:
                feature_vectors = w2v(doc_normalized, bkgd_docs_normalized, w2v_model, features['w2v'], feature_vectors)

            if 'cnn' in features:
                feature_vectors = run_cnn(normalize.xml_normalize(doc_raw), normalize.xml_normalize(bkgd_text_raw), tf_session)

            if 'wordonehot' in features:
                feature_vectors = wordonehot(doc_raw, bkgd_text_raw, full_vocab, features['wordonehot'], feature_vectors)

            # Save features and label
            feature_vectors = np.concatenate(feature_vectors, axis=0).astype(dtype)
            # Fail catastrphically on zero vector (not sure if we need this)
            #assert not (feature_vectors < 0.0001).all() 
            data.append(feature_vectors)
        if case["novelty"]:
            labels.append(1)
        else:
            labels.append(0)
        
        # save to HDF5 if desired
        if hdf5_path is not None and len(data) % hdf5_save_frequency == 0:
            with h5py.File(hdf5_path, 'a') as h5:
                data_np = np.array(data)
                labels_np = np.reshape(np.array(labels), (-1, 1))
                add_to_hdf5(h5, data_np, data_key)
                add_to_hdf5(h5, labels_np, labels_key, np.uint8)
                labels = list()
                data = list()
    # Save off any remainder
    if hdf5_path is not None and len(data) > 0:
        with h5py.File(hdf5_path, 'a') as h5:
            data_np = np.array(data)
            labels_np = np.reshape(np.array(labels), (-1, 1))
            add_to_hdf5(h5, data_np, data_key)
            add_to_hdf5(h5, labels_np, labels_key, np.uint8)

    mem_net_features['inputs'] = inputs
    mem_net_features['questions'] = questions
    mem_net_features['input_masks'] = input_masks
    mem_net_features['answers'] = labels
    
    ids = ["C" + str(clusterid) + "_P" + str(postid) for clusterid, postid in zip(clusterids,postids)]

   
    if 'mem_net' in features: 
        return mem_net_features, labels, ids
    if hdf5_path is not None:
        return hdf5_path, hdf5_path, ids
    else:
        return data, labels, ids

def add_to_hdf5(h5, data, label,dtype=np.float32):
    if label not in h5.keys():
        data_h5 = h5.create_dataset(label, data=data, maxshape=(None, data.shape[1]), dtype=dtype, compression='gzip')
    else:
        data_h5 = h5[label]
        data_h5_size = data_h5.shape[0] + data.shape[0]
        data_h5.resize(data_h5_size, axis=0)
        data_h5[-len(data):] = data

def main(all_clusters, lookup_order, document_data, features, parameters, vocab, full_vocab, encoder_decoder, lda_model, tf_session, w2v_model, hdf5_path=None, hdf5_save_frequency=100):
    '''
    Controls the generation of observations with the specified features.

    Args:
        argv (list): contains a set of all the cluster IDs, a dictionary of the document arrival order, an array of parsed JSON documents, the filename of the corpus, the feature tuple with the specified features, the vocabluary of the dataset and the skipthoughts vectors encoder/decoder

    Returns:
        list: contains for each obeservation
    '''
    data, labels, ids = gen_observations(all_clusters, lookup_order, document_data, features, parameters, vocab, full_vocab, encoder_decoder, lda_model, tf_session, w2v_model, hdf5_path, hdf5_save_frequency)

    return data, labels, ids
