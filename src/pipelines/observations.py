#!/usr/bin/env python

import sys
from src.utils import normalize
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from scipy import spatial
#import json
from collections import namedtuple, defaultdict
import os.path
from os.path import basename

def filter_text(doc):
   
    '''
    Purpose - to clean and normalize text
    Input - a string of text
    Output - the cleaned up version of that string
    '''
    
    clean_text = normalize.text_to_words(doc)
    return clean_text

def gen_observations(allClusters, lookupOrder, documentData, filename, features, vocab):

    '''
    Purpose - generate observations for each cluster found in JSON file and calculate the desired features
    Input - A set of cluster IDs, a dict of document arrival order, an array of the parsed JSON document data, the name of the corpus file, the desired features and the vocabulary
    Output - Array of namedtuples containing cluster_id, post_id, novelty, tfidf score, cosine similarity score (scores are None if feature is unwanted)
    '''
    
    # Prepare to store results of similarity assessments
    postScores = []
    postTuple = namedtuple('postScore','corpus,cluster_id,post_id,novelty,bagwordsScore,tfidfScore')    

    ''' 
    Iterate through clusters found in JSON file, do similarity assessments, 
    build a rolling corpus from ordered documents for each cluster
    '''
    for cluster in allClusters:
        
        # Determine arrival order in this cluster
        sortedEntries = [x[1] for x in sorted(lookupOrder[cluster], key=lambda x: x[0])]
                          
        # Set corpus to first doc in this cluster and prepare to update corpus with new document vocabulary
        corpus = filter_text(documentData[sortedEntries[0]]["body_text"]) 
                  
        # Create a document array for TFIDF
        corpusArray = corpus.split()
        
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
            
            if features.tf_idf:
                # Calculate L1 normalized TFIDF summation as Novelty Score for new document against Corpus
                # Credit to http://cgi.di.uoa.gr/~antoulas/pubs/ntoulas-novelty-wise.pdf 
                corpusArray.append(' '.join(rawdoc))
                vectorizer = TfidfVectorizer(norm=None)            
                tfidf = vectorizer.fit_transform(corpusArray)
                vectorValues = tfidf.toarray()
                tfidfScore = np.sum(vectorValues[-1])/docLength
            
            if features.cosine:  
                # Initialize the "CountVectorizer" object, which is scikit-learn's
                # bag of words tool.
                #http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
                vectorizer = CountVectorizer(analyzer = "word",  \
                                             vocabulary = vocab)

                # Combine Bag of Words dicts in vector format, calculate cosine similarity of resulting vectors  
                bagwordsVectors = (vectorizer.transform([doc, corpus])).toarray()
                similarityScore = 1 - spatial.distance.cosine(bagwordsVectors[0], bagwordsVectors[1])

            # Save results in namedtuple and add to array
            postScore = postTuple(corpusName, cluster, documentData[index]["post_id"], documentData[index]["novelty"], similarityScore, tfidfScore)
            postScores.append(postScore)

            # Update corpus
            corpus+=doc
 
    return postScores


def main(argv):
    allClusters, lookupOrder, documentData, file_name, features, vocab = argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]
    
    # Assess similarity based on document/corpus vectors and bag of words cosine similarity
    scores = gen_observations(allClusters, lookupOrder, documentData, file_name, features, vocab)
#     print("\n\nSimilarity and Novelty Scoring of a Document vs Corpus")
#     for score in scores: 
#         print("\n\nPost ID:", score.post_id)
#         print("Novelty Label:", score.novelty)
#         print("Bag of Words Similarity Score (1 = Identical, 0 = Completely Different):", score.bagwordsScore)
#         print("TFIDF Novelty Score (higher value indicates a larger difference):", score.tfidfScore)
    
    return scores