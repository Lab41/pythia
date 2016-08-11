"""
Constructing and loading dictionaries
"""
import numpy
from collections import OrderedDict
import pickle

def build_dictionary(text):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    wordcount = OrderedDict()
    for cc in text:
        words = cc.split()
        for w in words:
            wordcount[w] = wordcount.get(w,0) + 1

    ####
    words = list(wordcount.keys())
    freqs = list(wordcount.values())
    sorted_idx = numpy.argsort(freqs)[::-1]

    worddict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        worddict[words[sidx]] = idx+2 # 0: <eos>, 1: <unk>
    ###
        
    #Alternative: OrderedDict((word,idx + 2) for idx,(word,count) in enumerate(sorted(wordcount.items(),key=lambda _:_[1],reverse=True)))
    
    return worddict, wordcount

def load_dictionary(loc='/ais/gobi3/u/rkiros/bookgen/book_dictionary_large.pkl'):
    """
    Load a dictionary
    """
    with open(loc, 'rb') as f:
        worddict = pickle.load(f)
    return worddict

def save_dictionary(worddict, wordcount, loc):
    """
    Save a dictionary to the specified location 
    """
    with open(loc, 'wb') as f:
        pickle.dump(worddict, f)
        pickle.dump(wordcount, f)


