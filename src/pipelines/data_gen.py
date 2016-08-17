from src.featurizers.skipthoughts import skipthoughts as sk
from src.utils import normalize, tokenize
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
        words = normalize.remove_stop_words(tokenize.word_punct_tokens(sentence))
        if len(sentence_mask)>0: prev_mask = sentence_mask[-1]
        else: prev_mask = -1
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
                wordvector = np.random.uniform(0.0,1.0,(w2v_params['size'],))
            if wordvector is not None:
                wordvectorarray.append(wordvector)

    if mask_mode=='sentence': mask = sentence_mask
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

def wordonehot(doc, corpus, vocab, transformations, feature, min_length=None, max_length=None):

    norm_doc = normalize.xml_normalize(doc)
    norm_corpus = normalize.xml_normalize(corpus)
    doc_onehot = run_onehot(norm_doc, vocab, min_length, max_length)
    corpus_onehot = run_onehot(norm_corpus, vocab, min_length, max_length)
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

def gen_mem_net_observations(raw_doc, raw_corpus, sentences_full, mem_net_params, vocab, w2v_model, encoder_decoder):

    #use the specified mask mode where available
    if mem_net_params.get('mask_mode', False):
        mask_mode = mem_net_params["mask_mode"]
    else: mask_mode = 'sentence'

    if mem_net_params.get('st_embed', False):
        embed_mode = mem_net_params['embed_mode']
    else: embed_mode = 'word2vec'

    if mem_net_params.get('st_embed', False):
        corpus_vectors = sk.encode(encoder_decoder, sentences_full)
        doc_vectors = sk.encode(encoder_decoder, raw_doc)

        # Since each entry is a sentence, we use the index of each entry for the mask
        doc_masks = [index for index, w in enumerate(doc_vectors)]
        doc_questions = corpus_vectors
        doc_input = doc_vectors


    elif embed_mode == 'onehot':
        min_length = None
        max_length = None
        if mem_net_params.get('onehot_min_len', False):
            min_length = mem_net_params['onehot_min_len']
        if mem_net_params.get('onehot_max_len', False):
            max_length = mem_net_params['onehot_max_len']
        if mem_net_params.get('wordonehot_vocab', False):
            onehot_vocab = mem_net_params['onehot_vocab']
        else: onehot_vocab=vocab
        corpus_vectors = run_onehot(normalize.xml_normalize(raw_corpus), onehot_vocab, min_length, max_length)
        doc_vectors = run_onehot(normalize.xml_normalize(raw_doc), onehot_vocab, min_length, max_length)

        doc_questions = corpus_vectors.T
        doc_input = doc_vectors.T

        if mask_mode=='sentence':
            doc_masks = [index for index, w in enumerate(doc_input)] #TODO fix to be sentence somehow...
        else: doc_masks = [index for index, w in enumerate(doc_input)]


    elif embed_mode == 'word2vec':
        corpus_vectors, _ = run_w2v_matrix(w2v_model, raw_corpus, mem_net_params, mask_mode)
        doc_vectors, doc_masks = run_w2v_matrix(w2v_model, raw_doc, mem_net_params, mask_mode)

        if len(corpus_vectors)>0 and len(doc_vectors)>0:
            doc_questions = corpus_vectors
            doc_input = doc_vectors

    return doc_input, doc_questions, doc_masks

def gen_observations(all_clusters, lookup_order, documentData, features, parameters, vocab, encoder_decoder, lda_model, tf_session, w2v_model):
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
    mem_net_features = {}
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

        raw_corpus = first_doc

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
                doc_input, doc_questions, doc_masks = gen_mem_net_observations(raw_doc, raw_corpus, sentences_full, features['mem_net'], vocab, w2v_model, encoder_decoder)


                # Now add all of the input docs to the primary list if
                inputs.append(doc_input)
                questions.append(doc_questions)
                input_masks.append(doc_masks)

                sentences_full += tokenize.punkt_sentences(raw_doc)
            else:

                feature = list()

                if 'bow' in features:
                    feature = bow(doc, corpus, corpus_array, vocab, features['bow'], feature)

                if 'st' in features:
                    feature = st(raw_doc, sentences, encoder_decoder, features['st'], feature)

                if 'lda' in features:
                    feature = lda(doc, corpus, vocab, lda_model, features['lda'], feature)

                if 'w2v' in features:
                    feature = w2v(raw_doc, corpus, w2v_model, features['w2v'], feature)

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
            raw_corpus+=raw_doc
            sentences += get_first_and_last_sentence(raw_doc)

    mem_net_features['inputs'] = inputs
    mem_net_features['questions'] = questions
    mem_net_features['input_masks'] = input_masks
    mem_net_features['answers'] = labels

    if 'mem_net' in features:
        return mem_net_features, labels
    else:
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
