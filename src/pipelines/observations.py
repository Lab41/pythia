#!/usr/bin/env python
'''
Generates observations including specified features and novelty tags.
'''

from src.featurizers import skipthoughts as sk
from src.utils import normalize, tokenize
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from scipy import spatial
from collections import namedtuple, defaultdict
import os.path
from os.path import basename


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


def tfidf_sum(doc, corpus_array, vocab):
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
    vectorizer = TfidfVectorizer(norm=None, vocabulary=vocab)
    tfidf = vectorizer.fit_transform(corpus_array)
    vector_values = tfidf.toarray()
    tfidf_score = np.sum(vector_values[-1]) / doc_length
    return tfidf_score


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
    corpus_vector = np.mean(corpus_vectors, axis=0)
    doc_vector = np.mean(sk.encode(encoder_decoder, get_first_and_last_sentence(doc)), axis=0)
    skipthoughts = np.concatenate((doc_vector, corpus_vector), axis=0)
    return skipthoughts


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
    # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vectorizer = CountVectorizer(analyzer="word",
                                 vocabulary=vocab)

    # Combine Bag of Words dicts in vector format, calculate cosine similarity of resulting vectors
    bagwordsVectors = (vectorizer.transform([doc, corpus])).toarray()
    return bagwordsVectors


def run_lda(lda, doc, vocab):
    '''
    Calculates a vector of topic probabilities for a single document based on a trained LDA model.

    Args:
        lda (LatentDirichletAllocation): A LDA model previously fit to vocabulary of training data
        doc (str): the text (normalized and without stop words) of the document
        vocab (dict): the vocabulary of the data set

    Returns:
        array: a vector of topic probabilities based on a trained LDA model
    '''

    vectorizer = CountVectorizer(analyzer="word", vocabulary=vocab)
    docvector = vectorizer.transform([doc])
    return lda.transform(docvector)[0]


def gen_observations(all_clusters, lookup_order, documentData, filename, features, vocab, encoder_decoder, lda):
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
    postScores = []
    postTuple = namedtuple('postScore', 'corpus,cluster_id,post_id,novelty,bagwordsScore,tfidfScore,bog,skipthoughts,ldavector')

    # Iterate through clusters found in JSON file, do feature assessments,
    # build a rolling corpus from ordered documents for each cluster
    for cluster in all_clusters:
        # Determine arrival order in this cluster
        sortedEntries = [x[1] for x in sorted(lookup_order[cluster], key=lambda x: x[0])]

        first_doc = documentData[sortedEntries[0]]["body_text"]

        # Set corpus to first doc in this cluster and prepare to update corpus with new document vocabulary
        corpus = normalize.normalize_and_remove_stop_words(first_doc)

        # Create a document array for TFIDF
        corpus_array = [corpus]

        # Store a list of sentences in the cluster at each iteration
        sentences = []
        sentences += (get_first_and_last_sentence(first_doc))

        # Use filename as corpus name if corpus name was not defined in JSON
        try:
            corpusName = documentData[sortedEntries[0]]["corpus"]
        except KeyError:
            corpusName = basename(filename)

        for index in sortedEntries[1:]:
            # Find next document in order
            raw_doc = documentData[index]["body_text"]

            # normalize and remove stop words from doc
            doc = normalize.normalize_and_remove_stop_words(raw_doc)

            corpus_array.append(doc)

            similarityScore = None
            tfidfScore = None
            bog = None
            skipthoughts = None
            ldavector = None

            if features.tfidf_sum:
                tfidfScore = tfidf_sum(doc, corpus_array, vocab)

            if features.cos_similarity:
                bagwordsVectors = bag_of_words_vectors(doc, corpus, vocab)
                similarityScore = 1 - spatial.distance.cosine(bagwordsVectors[0], bagwordsVectors[1])

            if features.bag_of_words:
                bagwordsVectors = bag_of_words_vectors(doc, corpus, vocab)
                bog = np.concatenate(bagwordsVectors, axis=0)

            if features.skipthoughts:
                skipthoughts = skipthoughts_vectors(raw_doc, sentences, encoder_decoder)
                # Add newest sentence to sentences vector
                sentences += get_first_and_last_sentence(doc)

            if features.lda:
                doclda = run_lda(lda, doc, vocab)
                corpuslda = run_lda(lda, corpus, vocab)
                ldavector = np.concatenate([doclda, corpuslda], axis=0)

            # Save results in namedtuple and add to array
            postScore = postTuple(corpusName, cluster, documentData[index]["post_id"], documentData[index]["novelty"], similarityScore, tfidfScore, bog, skipthoughts, ldavector)
            postScores.append(postScore)

            # Update corpus
            corpus += doc

    return postScores


def main(argv):
    '''
    Controls the generation of observations with the specified features.

    Args:
        argv (list): contains a set of all the cluster IDs, a dictionary of the document arrival order, an array of parsed JSON documents, the filename of the corpus, the feature tuple with the specified features, the vocabluary of the dataset and the skipthoughts vectors encoder/decoder

    Returns:
        list: contains for each obeservation
    '''
    all_clusters, lookup_order, document_data, file_name, features, vocab, encoder_decoder, lda = argv
    observations = gen_observations(all_clusters, lookup_order, document_data, file_name, features, vocab, encoder_decoder, lda)

    return observations
