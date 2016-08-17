
# coding: utf-8

# ## Hard-coding data paths
# 

parsed_data_location = 'testing'
sample_location = 'data/stackexchange/anime'
file_extension = ".json"
num_pools = 8

# Import auxillary modules
import os
import json
import csv
import sys
from multiprocessing import Pool

# Import pythia modules
from src.utils import normalize, tokenize

# ## Tokenization and normalization
# 
# Who knows the best way to do this? I tried to match the expectations of both the skip-thoughts code and the pythia
# codebase as best I could.
# 
# For each document:
# 
# 1) Make list of sentences. We use utils.tokenize.punkt_sentences
# 
# 2) Normalize each sentence. Remove html and make everything lower-case. We use utils.normalize.xml_normalize
# 
# 3) Tokenize each sentence. Now each sentence is a string of space-separated tokens.
# We use utils.tokenize.word_punct_tokens and rejoin the tokens.
# 
# 
# 


# Instead of trying to parse in memory, can instead parse line by line and write to disk

def parse(path, fieldnames = ["body_text", "post_id","cluster_id", "order", "novelty"]):
    for line in open(path):
        temp_dict = json.loads(line)
        post_id = temp_dict['post_id']
        text = temp_dict['body_text']
        sentences = tokenize.punkt_sentences(text)
        normal = [normalize.xml_normalize(sentence) for sentence in sentences]
        tokens = [' '.join(tokenize.word_punct_tokens(sentence)) for sentence in normal]
        root, doc = os.path.split(path)
        base_doc = doc.split('.')[0]
        output_filename = "{}_{}.csv".format(base_doc, post_id)
        rel_path = os.path.relpath(root,sample_location)
        output_path = os.path.join(parsed_data_location, rel_path, output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok = True)
        with open(output_path, 'w') as token_file:
            writer = csv.DictWriter(token_file, fieldnames)
            writer.writeheader()
            output_dict = temp_dict
            for token in tokens:
                output_dict['body_text'] = token
                writer.writerow(output_dict)

fieldnames = ["body_text", "post_id","cluster_id", "order", "novelty"]
pool = Pool(num_pools)

pool.map(parse,(os.path.join(sample_location, root, doc)
                for root,dirs, files in os.walk(sample_location)
                for doc in files
                if doc.endswith(file_extension)))



