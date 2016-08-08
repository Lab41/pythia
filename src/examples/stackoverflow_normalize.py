
# coding: utf-8

# ## Hard-coding data paths
# 
# Mostly this notebook should just run. It however requires the user to fill in the cell below.
# 

# sample_location should be the path to a directory which contains your data. Each file should contain json-parsable lines. The directory can have subdirectories. The code will recursively find the files.

parsed_data_location = '/Users/chrisn/testing'
sample_location = '/Users/chrisn/mad-science/pythia/data/stackexchange'
file_extension = ".json"

# Import auxillary modules
import os
import json
import csv
import sys

# Import pythia modules
sys.path.append('/Users/chrisn/mad-science/pythia/')
from src.utils import normalize, tokenize

# ## Tokenization and normalization
# 
# Who knows the best way to do this? I tried to match the expectations of both the skip-thoughts code and the pythia codebase as best I could.
# 
# For each document:
# 
# 1) Make list of sentences. We use utils.tokenize.punkt_sentences
# 
# 2) Normalize each sentence. Remove html and make everything lower-case. We use utils.normalize.xml_normalize
# 
# 3) Tokenize each sentence. Now each sentence is a string of space-separated tokens. We use utils.tokenize.word_punct_tokens and rejoin the tokens.
# 
# 
# 


# Instead of trying to parse in memory, can instead parse line by line and write to disk
observed_paths = set() # used for logging purposes
fieldnames = ["body_text", "post_id","cluster_id", "order", "novelty"]
for root,dirs,files in os.walk(sample_location):
    for doc in files:
        if doc.endswith(file_extension):
            for line in open(os.path.join(sample_location,root,doc)):
                temp_dict = json.loads(line)
                post_id = temp_dict['post_id']
                text = temp_dict['body_text']
                sentences = tokenize.punkt_sentences(text)
                normal = [normalize.xml_normalize(sentence) for sentence in sentences]
                tokens = [' '.join(tokenize.word_punct_tokens(sentence)) for sentence in normal]
                base_doc = doc.split('.')[0]
                output_filename = "{}_{}.csv".format(base_doc,post_id)
                rel_path = os.path.relpath(root,sample_location)
                output_path = os.path.join(parsed_data_location,rel_path,output_filename)
                os.makedirs(os.path.dirname(output_path), exist_ok = True)
                with open(output_path,'w') as token_file:
                    #print(parsed_data_location,rel_path,output_filename)
                    writer = csv.DictWriter(token_file,fieldnames)
                    writer.writeheader()
                    output_dict = temp_dict
                    for token in tokens:
                        output_dict['body_text'] = token
                        writer.writerow(output_dict)


