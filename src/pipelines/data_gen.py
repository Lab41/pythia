from src.featurizers import skipthoughts as sk
from src.utils import normalize, tokenize
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder

import numpy as np
from scipy import spatial
from collections import defaultdict

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

def gen_observations(all_clusters, lookup_order, documentData, features, parameters, vocab, encoder_decoder, lda_topics, tf_session):
    '''
    Generates observations for each cluster found in JSON file and calculates the specified features.

    Args:
        all_clusters (set): cluster IDs
        lookup_order (dict): document arrival order
        documentData (array): parsed JSON documents
        filename (str): the name of the corpus file
        features (namedTuple): the specified features to be calculated
        vocab (dict): the vocabulary of the data set
        encoder_decoder (???): the encoder/decoder for skipthoughts vectors

    Returns:
        list: contains for each obeservation a namedtupled with the cluster_id, post_id, novelty, tfidf sum, cosine similarity, bag of words vectors and skip thoughts  (scores are None if feature is unwanted)
    '''

    # Prepare to store results of feature assessments
    data = list()
    labels = list()
    # mem_net_features is used when the mem_net algorithm is ran
    # It consist of inputs, labels(answers), input_masks and questions for each entry
    mem_net_features = list()
    inputs = []
    input_masks = []
    questions = []
    # Sentence punctuation delimiters
    punkt = ['.','?','!']

    #Iterate through clusters found in JSON file, do feature assessments,
    #build a rolling corpus from ordered documents for each cluster
    for cluster in all_clusters:
        # Determine arrival order in this cluster
        sortedEntries = [x[1] for x in sorted(lookup_order[cluster], key=lambda x: x[0])]

        first_doc = documentData[sortedEntries[0]]["body_text"]

        # Set corpus to first doc in this cluster and prepare to update corpus with new document vocabulary
        corpus = normalize.normalize_and_remove_stop_words(first_doc, **parameters)

        # Create a document array for TFIDF
        corpus_array = [corpus]

        # Store a list of sentences in the cluster at each iteration
        sentences = []
        sentences += (get_first_and_last_sentence(first_doc))
        sentences_full = tokenize.punkt_sentences(first_doc)

        for index in sortedEntries[1:]:
            # Find next document in order
            raw_doc = documentData[index]["body_text"]

            #normalize and remove stop words from doc
            doc = normalize.normalize_and_remove_stop_words(raw_doc, **parameters)

            corpus_array.append(doc)

            if 'mem_net' in features:
                mem_net_params = features['mem_net']
                #use the specified mask mode where available
                if mem_net_params.get('mask_mode', False):
                    mask_mode = mem_net_params["mask_mode"]
                else: mask_mode = 'sentence'

                if mem_net_params.get('st', False):
                    corpus_vectors = sk.encode(encoder_decoder, sentences_full)
                    doc_vectors = sk.encode(encoder_decoder, raw_doc)

                    sent_mask = [span[1]-1 for span in tokenize.punkt_sentence_span(raw_doc)]
                    input_masks.append(sent_mask)

                    inputs.append(corpus_vectors)
                    questions.append(doc_vectors)

                if mem_net_params.get('wordonehot', False):
                    min_length = None
                    max_length = None
                    if mem_net_params.get('wordonehot_min_len', False):
                        min_length = mem_net_params['one_hot_min_len']
                    if mem_net_params.get('wordonehot_max_len', False):
                        max_length = mem_net_params['one_hot_max_len']
                    if mem_net_params.get('wordonehot_vocab', False):
                        onehot_vocab = mem_net_params['wordonehot_vocab']
                    else: onehot_vocab=vocab
                    corpus_vectors = run_onehot(doc, onehot_vocab, min_length, max_length)
                    doc_vectors = run_onehot(doc, onehot_vocab, min_length, max_length)

                    #TODO transpose matrix and see if the matrix type is ok for the mem_net code

                if mem_net_params.get('word2vec', False):
                    corpus_vectors = run_w2v(corpus, vocab, min_length, max_length)
                    doc_vectors = run_w2v(doc, vocab, min_length, max_length)

                    inputs.append(corpus_vectors)
                    questions.append(doc_vectors)

                sentences_full += tokenize.punkt_sentences(raw_doc)

            feature = list()

            if 'bow' in features:
                feature = bow(doc, corpus, corpus_array, vocab, features['bow'], feature)

            if 'st' in features:
                feature = st(raw_doc, sentences, encoder_decoder, features['st'], feature)

            if 'lda' in features:
                feature = lda(doc, corpus, vocab, lda_topics, features['lda'], feature)

            if 'cnn' in features:
                feature = run_cnn(doc, corpus, tf_session)

            if 'wordonehot' in features:
                feature = wordonehot(doc, corpus, vocab, features['wordonehot'], feature)

            # Save feature and label
            feature = np.concatenate(feature, axis=0)
            data.append(feature)
            if documentData[index]["novelty"]: labels.append(1)
            else: labels.append(0)

            # Update corpus and add newest sentence to sentences vector
            corpus+=doc
            sentences += get_first_and_last_sentence(raw_doc)

    return data, labels

def main(argv):
    '''
    Controls the generation of observations with the specified features.

    Args:
        argv (list): contains a set of all the cluster IDs, a dictionary of the document arrival order, an array of parsed JSON documents, the filename of the corpus, the feature tuple with the specified features, the vocabluary of the dataset and the skipthoughts vectors encoder/decoder

    Returns:
        list: contains for each obeservation
    '''
    all_clusters, lookup_order, document_data, features, parameters, vocab, encoder_decoder, lda, tf_model = argv
    data, labels = gen_observations(all_clusters, lookup_order, document_data, features, parameters, vocab, encoder_decoder, lda, tf_model)

    return data, labels
