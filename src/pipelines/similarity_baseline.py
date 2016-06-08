import sys
import spacy
from spacy.attrs import *
from bs4 import BeautifulSoup
import urllib2
from sklearn.feature_extraction import DictVectorizer
from scipy import spatial

def getOnlyText(url):
   
   # Credit to http://glowingpython.blogspot.com/2014/09/text-summarization-with-nltk.html
   # Pull article text and strip out html encoding
   page = urllib2.urlopen(url).read().decode('utf8')
   soup = BeautifulSoup(page,"lxml")
   text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
   
   return text

def filterText(doc, nlp):
   
   # Remove stop words and non-alpha tokens from vocabulary   
   newVocab = []
   for token in doc:
      if not token.is_stop and token.is_alpha: newVocab.append(token.orth_)
   filteredDoc = nlp(' '.join(newVocab))

   return filteredDoc 

def main(argv):

   # Load Spacy's English tokenizer model
   print "Loading Spacy's English model"
   nlp = spacy.load('en')
   
   # Download two news articles from internet, supplied as urls in command line arguments
   utext1 = getOnlyText(argv[0])
   utext2 = getOnlyText(argv[1])
   
   # Use Spacy to tokenize documents, then remove stop words and non-alpha  
   print "Parsing files" 
   doc1 = filterText(nlp(utext1), nlp)      
   doc2 = filterText(nlp(utext2), nlp)

   # Similarity is estimated using the cosine metric, between Span.vector and other.vector. 
   # By default, Span.vector is computed by averaging the vectors of its tokens. [spacy.io]
   print "Document Vectors Similarity Score:", doc1.similarity(doc2)
    
   # Build Bag of Words with Spacy
   wordBag1 = doc1.count_by(LOWER)
   wordBag2 = doc2.count_by(LOWER)
   
   # Combine Bag of Words dicts in vector format, calculate cosine similarity of resulting vectors  
   vect = DictVectorizer(sparse=False)
   wordbagVectors = vect.fit_transform([wordBag2, wordBag1])
   print "Bag of Words Cosine Similarity Score:", 1 - spatial.distance.cosine(wordbagVectors[0], wordbagVectors[1])   

if __name__ == '__main__':
   if len(sys.argv) < 3:
      print "Usage: similarity_baseline.py file1 file2\n\nCompute cosine similarity between two documents"
      quit()
   main(sys.argv[1:])
