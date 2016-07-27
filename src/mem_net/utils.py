# This class provides utilities for processing data through a Dynamic Memory Network

import os as os
import numpy as np
import json
import re
import random
from src.utils.normalize import normalize_and_remove_stop_words

from bs4 import BeautifulSoup
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

# init_data using folder name processes the files from the file system.
# This module processes the data into tasks that can be used for training.
# The data is split into information blocks, question blocks, and answer blocks to process
# through the neural network. Returns the data as a list of dictionaries.
def init_data(fname, seed):
    print("==> Loading test from %s" % fname)
    train_tasks = [] # training list that will be returned
    test_tasks = [] # training list that will be returned

    num_files = 0

    random.seed(seed)
    for f in os.listdir(fname):
        documents = ""
        inData = open(fname + "/" + f)
        num_files += 1
        if random.random()>0.2:
            train_tasks.extend(parse_file(inData, f))
        else:
            test_tasks.extend(parse_file(inData, f))

    #documents = ""
    return train_tasks, test_tasks

def parse_file(inData, f):
    documents = ""
    tasks = []
    for i, line in enumerate(inData):
        #print(i, line)
        line = line.strip()
        try:
            post = json.loads(line) # make sure we can parse the json
        except Exception:
            print("Error with file " +  f)
            #continue
        text = post["body_text"]
        text = normalize_and_remove_stop_words(text) # call function from pythia normalize
        novelty = post["novelty"]
        task = {"C": "","Q": "", "A": ""}
        if i < 1:
            documents += text # add the first document before setting any tasks
        elif i < 200:
            task["C"] += documents # add the next 200 documents as a task with the new document as a question.
            task["Q"] = text
            task["A"] = novelty
            tasks.append(task.copy())
            documents += text
    return tasks

# Go fetch and process the raw data from the file system using init_data. See init_data.
def get_raw_data(directory, seed):
    raw_data_train, raw_data_test = init_data(directory, seed)
    return raw_data_train, raw_data_test

# Load glove data for word2vec            
def load_glove(dim):
    word2vec = {}
    
    print("==> loading glove")
    with open("data/glove/glove.6B." + str(dim) + "d.txt") as f:
        for line in f:    
            l = line.split()
            word2vec[l[0]] = list(map(float, l[1:]))
            
    print("==> glove is loaded")
    
    return word2vec


def create_vector(word, word2vec, word_vector_size, silent=False):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0,1.0,(word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print("utils.py::create_vector => %s is missing" % word)
    return vector

# Create a vector for these words if they are not in word2vec.
def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=False):
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab: 
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word
    
    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")


def get_norm(x):
    x = np.array(x)
    return np.sum(x * x)


def get_one_hot_doc(txt, char_vocab, replace_vocab=None, replace_char=' ',
                min_length=10, max_length=300, pad_out=True,
                to_lower=True, reverse = True,
                truncate_left=False, encoding=None):
    clean_txt = normalize(txt, replace_vocab, replace_char, min_length, max_length, pad_out,
                          to_lower, reverse, truncate_left,encoding)

    return to_one_hot(clean_txt, char_vocab)


zhang_lecun_vocab=list("abcdefghijklmnopqrstuvwxyz0123456789")
def to_one_hot(txt, vocab=zhang_lecun_vocab):
    vocab_hash = {b: a for a, b in enumerate(vocab)}
    vocab_size = len(vocab)
    one_hot_vec = np.zeros((1, vocab_size, len(txt)), dtype=np.float32)
    # run through txt and "switch on" relevant positions in one-hot vector
    for idx, char in enumerate(txt):
        try:
            vocab_idx = vocab_hash[char]
            one_hot_vec[0, vocab_idx, idx] = 1
        # raised if character is out of vocabulary
        except KeyError:
            pass
    return one_hot_vec

def normalize(txt, vocab=None, replace_char=' ',
                min_length=10, max_length=300, pad_out=True,
                to_lower=True, reverse = True,
                truncate_left=False, encoding=None):

    # store length for multiple comparisons
    txt_len = len(txt)

#     # normally reject txt if too short, but doing someplace else
#     if txt_len < min_length:
#         raise TextTooShortException("Too short: {}".format(txt_len))
    # truncate if too long
    if truncate_left:
        txt = txt[-max_length:]
    else:
        txt = txt[:max_length]
    # change case
    if to_lower:
        txt = txt.lower()
    # Reverse order
    if reverse:
        txt = txt[::-1]
    # replace chars
    if vocab is not None:
        txt = ''.join([c if c in vocab else replace_char for c in txt])
    # re-encode text
    if encoding is not None:
        txt = txt.encode(encoding, errors="ignore")
    # pad out if needed
    if pad_out and max_length>txt_len:
        txt = replace_char * (max_length - txt_len) + txt
    return txt