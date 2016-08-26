import copy

from src.featurizers.skipthoughts import skipthoughts as sk
from src.utils import normalize, tokenize, sampling
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder

import numpy as np
from scipy import spatial

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
        feature_vector.append(np.array([similarity]))
    return feature_vector

def bow(doc, corpus, corpus_array, vocab, bow, feature):
    vectors = bag_of_words_vectors(doc, corpus, vocab)
    feature = gen_feature(vectors, bow, feature)
    if 'tfidf' in bow:
        feature = tfidf_sum(doc, corpus_array, vocab, feature)
    return feature

def bag_of_words_vectors(doc, corpus, vocab):
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
                                 vocabulary = vocab)

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
    vectorizer = TfidfVectorizer(norm=None, vocabulary = vocab)
    tfidf = vectorizer.fit_transform(corpus_array)
    vector_values = tfidf.toarray()
    tfidf_score = np.sum(vector_values[-1])/doc_length
    feature.append(np.array([tfidf_score]))
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

def w2v(doc, corpus, w2v_model, w2v, feature):
    '''
     Calculates Word2Vec features for a document and corpus

     Args:
         doc (str): the text of the document (before any preprocessing)
         corpus (list): the text of the corpus
         w2v_model (gensim.Word2Vec): Trained Word2Vec model
          w2v (dict): Dictionary of Word2Vec parameters as set in master_pipeline. The dictionary
           will include keys for the model building parameters min_count, window, size, workers and pretrained.
           The dict may also have optional boolean keys for the feature operations append, difference, product and cos.
         feature (list): List of features extracted from text

     Returns:
         feature: List of features extracted from text
     '''

    docw2v = run_w2v(w2v_model, doc, w2v)
    corpusw2v = run_w2v(w2v_model, corpus, w2v)
    vectors = [docw2v, corpusw2v]
    feature = gen_feature(vectors, w2v, feature)
    return feature

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
          documentvector (list): List of Word2Vec vectors averaged across sentences
      '''

    # Get first and last sentences of document, break down sentences into words and remove stop words
    sentences = get_first_and_last_sentence(doc)
    normalizedsentences = []

    for sentence in sentences:
        words = normalize.remove_stop_words(tokenize.word_punct_tokens(sentence))
        normalizedsentences.append(words)

    wordvectorarray = []
    sentencevectorarray = []

    # Look up word vectors in trained Word2Vec model and build array of word vectors and sentence vectors
    for phrase in normalizedsentences:
        for word in phrase:
            wordvector = None
            try:
                wordvector = w2v_model[word]
            except:
                pass
            if wordvector is not None: wordvectorarray.append(wordvector)

        # Only calculate mean and append to sentence vector array if one or more word vectors were found
        if len(wordvectorarray) > 0:
            sentencevectorarray.append(np.mean(wordvectorarray, axis=0))

    # Only calculate mean if one or more sentences were added to sentence vector array, otherwise return array of zeroes
    if len(sentencevectorarray) > 0:
        documentvector =  np.mean(sentencevectorarray, axis=0)
    else:
        documentvector = np.zeros(w2v['size'])
    return documentvector

def run_cnn(doc, corpus, tf_session):
    doc_cnn, corpus_cnn = tf_session.transform_doc(doc, corpus)

    return [doc_cnn, corpus_cnn]

def wordonehot(doc, corpus, vocab, transformations, feature, min_length=None, max_length=None):
    # TODO: do we need to normalize here too?
    doc_array = tokenize.word_punct_tokens(doc)
    corpus_array = tokenize.word_punct_tokens(corpus)
    doc_onehot = run_onehot(doc, vocab, min_length, max_length)
    corpus_onehot = run_onehot(corpus, vocab, min_length, max_length)
    feature = gen_feature([doc_onehot, corpus_onehot], transformations, feature)
    return feature

def run_onehot(doc, vocab, min_length=None, max_length=None):
    """ One-hot encode array of tokens, given a vocabulary mapping
    them to 0-to-n integer space

    Args:
        doc (list): list of tokens; should correspond to the keys in vocab (so,
            typically str)
        vocab (dict): map of vocab items to integers (zero based)
        min_length: if not None, enforce a minimum document length by zero-padding
            the right edge of the result
        max_length: if not None, truncate documents to max_length

    Returns:
        NDArray (vocab size, doc length), with 1 indicating presence of vocab item
            at that position. Out-of-vocab entries do not appear in the result.
    """
    # transform only the non-null entries in the document
    doc_indices = [doc_idx for doc_idx in
                        [ vocab.get(wd, None) for wd in doc ]
                        if doc_idx is not None]
    vocab_size = len(vocab)
    doc_length = len(doc_indices)
    doc_onehot = np.zeros((vocab_size, doc_length), dtype=np.float32)
    for token_idx, token in enumerate(doc_indices):
        doc_onehot[token, token_idx] = 1

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

def gen_observations(all_clusters, lookup_order, document_data, features, parameters, vocab, encoder_decoder, lda_model, tf_session, w2v_model, random_state=np.random):
    '''
    Generates observations for each cluster found in JSON file and calculates the specified features.

    Args:
        all_clusters (set): cluster IDs
        lookup_order (dict): document arrival order
        document_data (array): parsed JSON documents
        features (namedTuple): the specified features to be calculated
        parameters (???): data structure with run parameters
        vocab (dict): the vocabulary of the data set
        encoder_decoder (???): the encoder/decoder for skipthoughts vectors
        lda_model (sklearn.???): trained LDA model
        tf_session: active TensorFlow session
        w2v_model (gensim.word2vec): trained word2vec model

    Returns:
        list, list: lists containing featurized data (np.array) and associated 
            labels (int), 0 if duplicate and 1 if novel
    '''

    # Prepare to store results of feature assessments
    data = list()
    labels = list()
    corpus_unprocessed = list()

    # Iterate through clusters found in JSON file, generate observations
    # pairing data and label
    for cluster in all_clusters:
        # Determine arrival order in this cluster
        sorted_entries = [x[1] for x in sorted(lookup_order[cluster], key=lambda x: x[0])]

        observations = [document_data[sorted_entries[0]]]

        for index in sorted_entries[1:]:
            next_doc = document_data[index]
            observations.append(next_doc)
            corpus_unprocessed.append(observations)
    
    # Resample if necessary
    # If oversampling +/- replacement, sample up
    # to larger class size for both classes, with replacement
    # If -oversampling, sample down to 
    # smaller class size for both classes with or w/o replacement
    if 'resampling' in parameters:
        if 'over' in parameters:
            desired_size = None
            parameters['replacement'] = True
        else:
            desired_size = -np.Inf
        if 'replacement' in parameters:
            replacement = True
        else:
            replacement = False
        corpus = sampling.label_sample(corpus_unprocessed, "novelty", replacement, desired_size, random_state)  
    else:
        corpus = corpus_unprocessed

    # Featurize each observation
    # Some duplication of effort here bc docs will appear multiple times 
    # across observations
    for case in corpus:
        # Create raw and normalized document arrays
        case_docs_raw = [ record['body_text'] for record in case['data'] ]
        case_docs_normalized = [ normalize.normalize_and_remove_stop_words(body_text) for body_text in case_docs_raw ]
        # Pull out query documents
        doc_raw = case_docs_raw[-1]
        doc_normalized = case_docs_raw[-1]
        # Create lists of background documents
        bkgd_docs_raw = case_docs_raw[:-1]
        bkgd_docs_normalized = case_docs_normalized[:-1]
        bkgd_text_raw = '\n'.join(bkgd_docs_raw)

        feature_vectors = list()

        if 'bow' in features:
            feature_vectors = bow(doc_normalized, bkgd_text_raw, bkgd_docs_normalized, vocab, features['bow'], feature_vectors)

        if 'st' in features:
            sentences = [ get_first_and_last_sentence(doc) for doc in bkgd_raw ]
            feature_vectors = st(doc_raw, sentences, encoder_decoder, features['st'], feature_vectors)

        if 'lda' in features:
            feature_vectors = lda(doc_normalized, bkgd_docs_normalized, vocab, lda_model, features['lda'], feature_vectors)

        if 'w2v' in features:
            feature_vectors = w2v(doc_raw, bkgd_docs_normalized, w2v_model, features['w2v'], feature_vectors)

        if 'cnn' in features:
            feature_vectors = run_cnn(doc_normalized, bkgd_docs_normalized, tf_session)

        if 'wordonehot' in features:
            feature_vectors = wordonehot(doc_raw, bkgd_text_raw, vocab, features['wordonehot'], feature_vectors)

        # Save features and label
        feature_vectors = np.concatenate(feature_vectors, axis=0)
        data.append(feature_vectors)
        if case[-1]["novelty"]: 
            labels.append(1)
        else: 
            labels.append(0)
        
    return data, labels

def main(argv):
    '''
    Controls the generation of observations with the specified features.

    Args:
        argv (list): contains a set of all the cluster IDs, a dictionary of the document arrival order, an array of parsed JSON documents, the filename of the corpus, the feature tuple with the specified features, the vocabluary of the dataset and the skipthoughts vectors encoder/decoder

    Returns:
        list: contains for each obeservation
    '''
    all_clusters, lookup_order, document_data, features, parameters, vocab, encoder_decoder, lda_model, tf_session, w2v_model = argv
    data, labels = gen_observations(all_clusters, lookup_order, document_data, features, parameters, vocab, encoder_decoder, lda_model, tf_session, w2v_model)

    return data, labels
