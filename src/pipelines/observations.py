#!/usr/bin/env python
'''
Generates observations including specified features and novelty tags.
'''

import sys
from src.featurizers import skipthoughts as sk
from src.utils import normalize
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from scipy import spatial
from collections import namedtuple, defaultdict
import os.path
from os.path import basename

def filter_text(doc):

    '''
    Purpose - to clean and normalize text
    Input - a string of text
    Output - the cleaned up version of that string
    '''

    clean_text = normalize.normalize_and_remove_stop_words(doc)
    return clean_text

def bag_of_words(doc, corpus, vocab):

    '''
    Creates bag of words vectors for doc and corpus for a given vocabulary.
    
    Args:
        doc (str): the text (normalized and without stop words) of the document
        corpus (str): the text (normalized and without stop words) of the corpus for that cluster
        vocab (dict): the vocabulary of the data set
        
    Returns:
        array: an array of the bag of words vectors
    '''

    # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
    #http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vectorizer = CountVectorizer(analyzer = "word",  \
                                 vocabulary = vocab)

    # Combine Bag of Words dicts in vector format, calculate cosine similarity of resulting vectors
    bagwordsVectors = (vectorizer.transform([doc, corpus])).toarray()
    return bagwordsVectors


def gen_observations(all_clusters, lookup_order, documentData, filename, features, vocab, encoder_decoder):

    '''
    Generates observations for each cluster found in JSON file and calculates the specified features.
    
    Args:
        all_clusters (set): a set of cluster IDs
        lookup_order (dict): a dict of document arrival order
        
    Input - A set of cluster IDs, a dict of document arrival order, an array of the parsed JSON document data, the name of the corpus file, the desired features and the vocabulary
    Output - Array of namedtuples containing cluster_id, post_id, novelty, tfidf sum, cosine similarity, bag of words vectors and skip thoughts  (scores are None if feature is unwanted)
    '''

    # Prepare to store results of similarity assessments
    postScores = []
    postTuple = namedtuple('postScore','corpus,cluster_id,post_id,novelty,bagwordsScore,tfidfScore,bog,skipthoughts')
    '''
    Iterate through clusters found in JSON file, do similarity assessments,
    build a rolling corpus from ordered documents for each cluster
    '''
    for cluster in all_clusters:

        # Determine arrival order in this cluster
        sortedEntries = [x[1] for x in sorted(lookup_order[cluster], key=lambda x: x[0])]

        # Set corpus to first doc in this cluster and prepare to update corpus with new document vocabulary
        corpus = filter_text(documentData[sortedEntries[0]]["body_text"])

        # Create a document array for TFIDF
        corpusArray = corpus.split()

        # Store a list of sentences in the cluster at each iteration
        sentences = []
        sentences.append(filter_text(documentData[sortedEntries[0]]["body_text"]))

        # Use filename as corpus name if corpus name was not defined in JSON
        try: corpusName = documentData[sortedEntries[0]]["corpus"]
        except KeyError: corpusName = basename(filename)

        for index in sortedEntries[1:]:

            # Find next document in order
            doc = filter_text(documentData[index]["body_text"])
            rawdoc = doc.split()
            docLength = len(rawdoc)

            similarityScore = None
            tfidfScore = None
            bog = None
            skipthoughts = None

            if features.tf_idf:
                # Calculate L1 normalized TFIDF summation as Novelty Score for new document against Corpus
                # Credit to http://cgi.di.uoa.gr/~antoulas/pubs/ntoulas-novelty-wise.pdf
                corpusArray.append(' '.join(rawdoc))
                vectorizer = TfidfVectorizer(norm=None)
                tfidf = vectorizer.fit_transform(corpusArray)
                vectorValues = tfidf.toarray()
                tfidfScore = np.sum(vectorValues[-1])/docLength

            if features.cosine:
                bagwordsVectors = bag_of_words(doc, corpus, vocab)
                similarityScore = 1 - spatial.distance.cosine(bagwordsVectors[0], bagwordsVectors[1])

            if features.bog:
                bagwordsVectors = bag_of_words(doc, corpus, vocab)
                bog = np.concatenate(bagwordsVectors, axis=0)

            if features.skipthoughts:
                # Add newest sentence to sentences vector
                # The encode function produces an array of skipthought vectors with as many rows as there were sentences and 4800 dimensions
                # See the combine-skip section of the skipthoughts paper for a detailed explanation of the array
                corpus_vectors = sk.encode(encoder_decoder, sentences)
                #print('coprus vectors: ', corpus_vectors)
                corpus_vector = np.mean(corpus_vectors, axis = 0)
                #print('coprus vector: ', corpus_vector)
                doc_vector = sk.encode(encoder_decoder, [doc])
                skipthoughts = np.concatenate((doc_vector[0], corpus_vector), axis=0)
                #print('observation: ', skipthoughts)
                sentences.append(doc)

            # Save results in namedtuple and add to array
            postScore = postTuple(corpusName, cluster, documentData[index]["post_id"], documentData[index]["novelty"], similarityScore, tfidfScore, bog, skipthoughts)
            postScores.append(postScore)

            # Update corpus
            corpus+=doc

    return postScores


def main(argv):
    all_clusters, lookupOrder, documentData, file_name, features, vocab, encoder_decoder = argv
    observations = gen_observations(all_clusters, lookupOrder, documentData, file_name, features, vocab, encoder_decoder)

    return observations
