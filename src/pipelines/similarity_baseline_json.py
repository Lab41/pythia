import sys
import spacy
from spacy.attrs import *
from sklearn.feature_extraction import DictVectorizer
from scipy import spatial
import json
from collections import namedtuple, defaultdict
from os.path import basename


def parse_json(fileName):
   
    '''
    Purpose - Parses a JSON file containing document data. 
    Input - File name with JSON data
    Output - A set of cluster IDs, a dictionary mapping to sets of tuples with document 
             arrival order and indices, an array of the parsed JSON document data
        
    The JSON data schema for ingesting documents into Pythia is:
    corpus = name of corpus 
    cluster_id = unique identifier for associated cluster
    post_id = unique identifier for element (document, post, etc)
    order = int signifying order item was received 
    body_text = text of element (document, post, etc)
    novelty = boolean assessment of novelty     
    '''
           
    documentData = []
    allClusters = set()
    lookupOrder = defaultdict(set)
    i = 0
        
    # Read JSON file line by line and retain stats about number of clusters and order of objects 
    with open(fileName,'r') as dataFile:
        for line in dataFile:
            parsedData = json.loads(fix_escapes(line))
            allClusters.add(parsedData["cluster_id"])            
            lookupOrder[parsedData["cluster_id"]].add((parsedData["order"],i))            
            documentData.append(parsedData)
            i += 1
    return allClusters, lookupOrder, documentData


def fix_escapes(line):
    
    # Remove embedded special left/right leaning quote characters in body_text segment of json object
    if line.find('\\\xe2\x80\x9d'):
        spot = line.find("body_text")
        line = line[:spot+13] + line[spot+13:].replace('\\\xe2\x80\x9d','\\"')
    if line.find('\\\xe2\x80\x9c'):
        spot = line.find("body_text")
        line = line[:spot+13] + line[spot+13:].replace('\\\xe2\x80\x9c','\\"')
    return line

def filter_text(doc, nlp):
   
    '''
    Purpose - Removes stop words, out of vocabulary words and non-alpha words from a 
              spacy document's vocabulary
    Input - A spacy document and the spacy English parser
    Output - A new spacy document
    '''
   
    # Remove stop words, out of vocabulary and non-alpha tokens from document vocabulary   
    newVocab = []
    for token in doc:
       if not token.is_stop and not token.is_oov and token.is_alpha: newVocab.append(token.orth_)
    filteredDoc = nlp(' '.join(newVocab))
    return filteredDoc 


def assess_similarity(allClusters, lookupOrder, documentData, nlp, filename):

    '''
    Purpose - Do vector comparison and bag of words cosine similarity for each cluster found in JSON file
    Input - A set of cluster IDs, a dict of document arrival order, an array of the parsed JSON document data, and a spacy English parser
    Output - Array of namedtuples containing cluster_id, post_id, novelty, vector score, cosine similarity score
    '''
    
    # Prepare to store results of similarity assessments
    postScores = []
    postTuple = namedtuple('postScore','corpus,cluster_id,post_id,novelty,vectorScore,bagwordsScore')    

    ''' 
    Iterate through clusters found in JSON file, do similarity assessments, 
    build a rolling corpus from ordered documents for each cluster
    '''
    for cluster in allClusters:
        
        # Determine arrival order in this cluster
        sortedEntries = [x[1] for x in sorted(lookupOrder[cluster], key=lambda x: x[0])]
                          
        # Set corpus to first doc in this cluster and prepare to update corpus with new document vocabulary
        corpus = filter_text(nlp(documentData[sortedEntries[0]]["body_text"]), nlp) 
        updateCorpus = []
        for token in corpus: updateCorpus.append(token.orth_)
        
        # Use filename as corpus name if corpus name was not defined in JSON
        try: corpusName = documentData[sortedEntries[0]]["corpus"]
        except KeyError: corpusName = basename(filename)
        
        # Insert first document into scoring array for recordkeeping with similarity scores as -1
        postScore = postTuple(corpusName, cluster, documentData[sortedEntries[0]]["post_id"], documentData[sortedEntries[0]]["novelty"], -1, -1)
        postScores.append(postScore)

        for index in sortedEntries[1:]:
                
            # Find next document in order
            doc = filter_text(nlp(documentData[index]["body_text"]), nlp)

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
            postScore = postTuple(corpusName, cluster, documentData[index]["post_id"], documentData[index]["novelty"], vectorScore, similarityScore)
            postScores.append(postScore)
            
            # Update corpus
            for token in doc: updateCorpus.append(token.orth_)
            corpus = nlp(' '.join(updateCorpus))
 
    return postScores


def main(argv):

    # Load Spacy's English tokenizer model
    print "Loading Spacy's English model. This can take a few seconds."
    nlp = spacy.load('en')
   
    # Parse JSON file that was supplied in command line argument
    allClusters, lookupOrder, documentData = parse_json(argv[0])
    
    # Assess similarity based on document/corpus vectors and bag of words cosine similarity
    scores = assess_similarity(allClusters, lookupOrder, documentData, nlp, argv[0])    
    for score in scores: print score
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
       print "Usage: similarity_baseline_json.py file1\n\nCompute bag of words cosine similarity between documents defined in JSON file (file1)"
    else: main(sys.argv[1:])
