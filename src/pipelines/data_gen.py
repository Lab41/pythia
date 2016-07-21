from src.featurizers import skipthoughts as sk
from src.utils import normalize, tokenize
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from scipy import spatial
from collections import defaultdict

def gen_feature(vectors, feature_dict, feature):
    if 'append' in feature_dict:
        feature.append(np.concatenate(vectors, axis=0))
    if 'difference' in feature_dict:
        feature.append(np.subtract(vectors[0], vectors[1]))
    if 'product' in feature_dict:
        feature.append(np.multiply(vectors[0], vectors[1]))
    if 'cos' in feature_dict:
        similarity = 1 - spatial.distance.cosine(vectors[0], vectors[1])
        feature.append(np.array([similarity]))
    return feature

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
        lda (LatentDirichletAllocation): A LDA model previously fit to vocabulary of training data
        doc (str): the text (normalized and without stop words) of the document
        vocab (dict): the vocabulary of the data set
        
    Returns:
        array: a vector of topic probabilities based on a trained LDA model
    '''

    vectorizer = CountVectorizer(analyzer = "word", vocabulary = vocab)
    docvector = vectorizer.transform([doc])  
    return lda_topics.transform(docvector)[0]

def gen_observations(all_clusters, lookup_order, documentData, features, vocab, encoder_decoder, lda_topics):
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

    #Iterate through clusters found in JSON file, do feature assessments,
    #build a rolling corpus from ordered documents for each cluster
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

        for index in sortedEntries[1:]:
            # Find next document in order
            raw_doc = documentData[index]["body_text"]
            
            #normalize and remove stop words from doc
            doc = normalize.normalize_and_remove_stop_words(raw_doc)

            corpus_array.append(doc)
            
            feature = list()

            if 'bow' in features:
                feature = bow(doc, corpus, corpus_array, vocab, features['bow'], feature)

            if 'st' in features:
                feature = st(raw_doc, sentences, encoder_decoder, features['st'], feature)

            if 'lda' in features:
                feature = lda(doc, corpus, vocab, lda_topics, features['lda'], feature)

            # Save feature and label
            feature = np.concatenate(feature, axis=0)
            data.append(feature)
            if documentData[index]["novelty"]: labels.append(1)
            else: labels.append(0)
           
            # Update corpus and add newest sentence to sentences vector
            corpus+=doc
            sentences += get_first_and_last_sentence(doc)

    return data, labels
            
def main(argv):
    '''
    Controls the generation of observations with the specified features.
    
    Args:
        argv (list): contains a set of all the cluster IDs, a dictionary of the document arrival order, an array of parsed JSON documents, the filename of the corpus, the feature tuple with the specified features, the vocabluary of the dataset and the skipthoughts vectors encoder/decoder
    
    Returns:
        list: contains for each obeservation
    '''
    all_clusters, lookup_order, document_data, features, vocab, encoder_decoder, lda = argv
    data, labels = gen_observations(all_clusters, lookup_order, document_data, features, vocab, encoder_decoder, lda)

    return data, labels