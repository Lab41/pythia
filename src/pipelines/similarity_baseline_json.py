import sys
import spacy
from spacy.attrs import *
from sklearn.feature_extraction import DictVectorizer
from scipy import spatial
import json
from collections import namedtuple

def parseJSON(fileName):
   
    '''
    Purpose - Parses a JSON file containing document data. 
    Input - File name with JSON data
    Output - A set of cluster IDs, a dict of document arrival order and an array of the parsed JSON document data
        
    The JSON data schema for ingesting documents into Pythia is:
    corpus = name of corpus 
    cluster_id = unique identifier for associated cluster
    post_id = unique identifier for element (document, post, etc)
    order = int signifying order item was received 
    body_text = text of element (document, post, etc)
    novelty = boolean assessment of novelty 
    
    Function returns an array of dictionaries containing data in JSON schema 
    '''
           
    documentData = []
    allClusters = set()
    lookupOrder = {}
    i = 0
    
    # Read JSON file line by line and retain stats about number of clusters and order of objects 
    with open(fileName,'r') as dataFile:
        for line in dataFile: 
            parsedData = json.loads(line)
            allClusters.add(parsedData["cluster_id"])
            lookupOrder[str(parsedData["cluster_id"]) + "_" + str(parsedData["order"])] = i
            documentData.append(parsedData)
            i += 1
                  
    return allClusters, lookupOrder, documentData

def filterText(doc, nlp):
   
    '''
    Purpose - Removes stop words, out of vocabulary words and non-alpha words from a spacy document's vocabulary
    Input - A spacy document and the spacy English parser
    Output - A new spacy document
    '''
   
    # Remove stop words, out of vocabulary and non-alpha tokens from document vocabulary   
    newVocab = []
    for token in doc:
       if not token.is_stop and not token.is_oov and token.is_alpha: newVocab.append(token.orth_)
    filteredDoc = nlp(' '.join(newVocab))

    return filteredDoc 

def assessSimilarity(allClusters, lookupOrder, documentData, nlp):

    '''
    Purpose - Do vector comparison and bag of words cosine similarity for each cluster found in JSON file
    Input - A set of cluster IDs, a dict of document arrival order, an array of the parsed JSON document data, and a spacy English parser
    Output - Array of namedtuples containing cluster_id, post_id, novelty, vector score, cosine similarity score
    '''
    
    # Store results of similarity assessments
    postScores = []
    postScore = namedtuple('postScore','cluster_id,post_id,novelty,vectorScore,bagwordsScore')    

    # Iterate through clusters found in JSON file, do similarity assessments, build a rolling corpus from ordered documents for each cluster
    for cluster in allClusters:
                
        # Set corpus to first doc in this cluster and prepare to update corpus with new document vocabulary
        corpus = filterText( nlp( documentData[ lookupOrder[ str(cluster)+"_0"] ] ["body_text"] ), nlp) 
        updateCorpus = []
        for token in corpus: updateCorpus.append(token.orth_)

        i=1
        search = True
        while search is True:
            if lookupOrder.has_key(str(cluster) + "_" + str(i)):
                
                # Find next document in order
                index =  lookupOrder[str(cluster) + "_" + str(i)]
                postID = documentData[index]["post_id"]
                novelty = documentData[index]["novelty"]
                doc = filterText(nlp(documentData[index]["body_text"]), nlp)

                # Document vs Corpus vector comparison
                vectorScore = doc.similarity(corpus) 

                # Build Bag of Words with Spacy
                docBagWords = doc.count_by(LOWER)
                corpusBagWords = corpus.count_by(LOWER)
                
                # Combine Bag of Words dicts in vector format, calculate cosine similarity of resulting vectors  
                vect = DictVectorizer(sparse=False)
                bagwordsVectors = vect.fit_transform([docBagWords, corpusBagWords])
                similarityScore = 1 - spatial.distance.cosine(bagwordsVectors[0], bagwordsVectors[1])
                
                # Save results in namedtuple and add to array
                postScore = (cluster, postID, novelty, vectorScore, similarityScore)
                postScores.append(postScore)
                
                # Update corpus
                for token in doc: updateCorpus.append(token.orth_)
                corpus = nlp(' '.join(updateCorpus))
                i += 1
                
            else: search = False
 
    return postScores

def main(argv):

    # Load Spacy's English tokenizer model
    print "Loading Spacy's English model. This can take a few seconds."
    nlp = spacy.load('en')
   
    # Parse JSON file that was supplied in command line argument
    allClusters, lookupOrder, documentData = parseJSON(argv[0])
    
    # Assess similarity based on document/corpus vectors and bag of words cosine similarity
    scores = assessSimilarity(allClusters, lookupOrder, documentData, nlp)    
    for score in scores: print score
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
       print "Usage: similarity_baseline_json.py file1\n\nCompute bag of words cosine similarity between documents defined in JSON file (file1)"
    main(sys.argv[1:])
