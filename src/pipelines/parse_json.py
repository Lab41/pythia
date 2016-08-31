#!/usr/bin/env python
'''
Parses Pythia-style JSON files and inventories most frequent words in the corpus.
'''
import sys
import json
import os.path
from os.path import basename
from collections import defaultdict, namedtuple, OrderedDict
import nltk
from nltk.tokenize import word_tokenize
import numpy

def parse_json(folder, seed=1, **kwargs):

    '''
    Purpose - Parses a folder full of JSON files containing document data.
    Input - a directory full of files with JSON data
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

    data = []
    clusters = set()
    order = defaultdict(set)
    wordcount = defaultdict(int)
    i = 0

    test_data = []
    test_clusters = set()
    test_order = defaultdict(set)
    j = 0

    random_state = numpy.random.RandomState(seed)
    for file_name in os.listdir (folder):
        if file_name.endswith(".json"):
            # Read JSON file line by line and retain stats about number of clusters and order of objects
            full_file_name = os.path.join(folder, file_name)

            with open(full_file_name,'r') as dataFile:
                if random_state.random_sample() > 0.2:
                    for line in dataFile:
                        parsedData = json.loads(fix_escapes(line))
                        clusters.add(parsedData["cluster_id"])
                        order[parsedData["cluster_id"]].add((parsedData["order"],i))
                        wordcount = count_vocab(parsedData["body_text"], wordcount)
                        data.append(parsedData)
                        i += 1
                else:
                    for line in dataFile:
                        parsedData = json.loads(fix_escapes(line))
                        test_clusters.add(parsedData["cluster_id"])
                        test_order[parsedData["cluster_id"]].add((parsedData["order"],j))
                        test_data.append(parsedData)
                        j += 1
    return clusters, order, data, test_clusters, test_order, test_data, wordcount


def fix_escapes(line):
    '''
    Purpose - Substitutes any leaning right/left quote characters in body_text segment of JSON files.
    Input - A line of text from a Pythia-style JSON file
    Output - The input line with any leaning right/left quote characters replaced with standard quotes        
    '''

    # Remove embedded special left/right leaning quote characters in body_text segment of json object
    if line.find('\\\xe2\x80\x9d'):
        spot = line.find("body_text")
        line = line[:spot+13] + line[spot+13:].replace('\\\xe2\x80\x9d','\\"')
    if line.find('\\\xe2\x80\x9c'):
        spot = line.find("body_text")
        line = line[:spot+13] + line[spot+13:].replace('\\\xe2\x80\x9c','\\"')
    return line

def count_vocab(text, wordcount):
    '''
    Purpose - Counts the number of times any word appears in a text string.
    Input - A line of text and a dictionary with words and their associated counts
    Output - An updated dictionary with words and their associated counts        
    '''

    # Tokenize text and add words to corpus dictionary
    wordlist = word_tokenize(text)
    for word in wordlist: wordcount[word] += 1

    return wordcount

def order_vocab(tokencount):
    '''
    Purpose - Determine the most frequently occurring words in the corpus
    Input - A dictionary with words and their associated counts
    Output - An Ordered dictionary, starting with the highest frequency words        
    '''

    # Determine descending order for word counts
    # Credit to https://github.com/ryankiros/skip-thoughts/
    words = list(tokencount.keys())
    freqs = list(tokencount.values())
    sorted_idx = numpy.argsort(freqs)[::-1]

    wordorder = OrderedDict()
    for idx, sidx in enumerate(sorted_idx): wordorder[words[sidx]] = idx+2
    return wordorder

def main(folder, parameters):

    print("parsing json files...",file=sys.stderr)
    
    #folder, parameters = argv
    
    # Parse JSON file that was supplied in command line argument
    clusters, order, data, test_clusters, test_order, test_data, wordcount = parse_json(folder, **parameters)

    # Determine descending order for words based on count
    rawwords = [i[0] for i in sorted(wordcount.items(), key=lambda x: (x[1], x[0]), reverse=True)]

    return clusters, order, data, test_clusters, test_order, test_data, rawwords

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: parse_json.py dir1\n\nparses data from JSON files defined in directory (dir1)")
    else:
        #FIXME: does this work?
        main(*sys.argv[1:])
