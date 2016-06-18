#!/usr/bin/env python

import sys
import json
#import os.path
from os.path import basename

def parse_json(folder):
   
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
           
    documentData = []
    allClusters = set()
    lookupOrder = defaultdict(set)
    i = 0
        
    for file_name in os.listdir (folder):
        #print "file: ", file_name
        if file_name.endswith(".json"):
            # Read JSON file line by line and retain stats about number of clusters and order of objects
            full_file_name = os.path.join(folder, file_name)
            with open(full_file_name,'r') as dataFile:
                #print "open"
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

def main(argv):
  
    print "parsing json files..."
    # Parse JSON file that was supplied in command line argument
    allClusters, lookupOrder, documentData = parse_json(argv[0])
    
    return allClusters, lookupOrder, documentData
        
if __name__ == '__main__':
    if len(sys.argv) < 2:
       print "Usage: parse_json.py dir1\n\nparses data from JSON files defined in directory (dir1)"
    else: 
        main(sys.argv[1:])