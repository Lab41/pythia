# This class provides utilities for processing data through a Dynamic Memory Network

import os as os
import numpy as np
import json
import re
import random
from src.utils.normalize import normalize_and_remove_stop_words, xml_normalize
from src.utils.tokenize import word_punct_tokens, punkt_sentences
from src.pipelines import data_gen
from src.featurizers import skipthoughts as sk

import theano
floatX = theano.config.floatX

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
        text = xml_normalize(text) # call function from pythia normalize
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

def analyze_clusters(all_clusters, lookup_order, documentData):
    tasks = []
    lil_spacy = " "
    #Iterate through clusters found in JSON file, do feature assessments,
    #build a rolling corpus from ordered documents for each cluster
    for cluster in all_clusters:
        # Determine arrival order in this cluster
        sortedEntries = [x[1] for x in sorted(lookup_order[cluster], key=lambda x: x[0])]

        first_doc = documentData[sortedEntries[0]]["body_text"]

        # Set corpus to first doc in this cluster and prepare to update corpus with new document vocabulary
        corpus = lil_spacy.join(word_punct_tokens(xml_normalize(first_doc)))

        # #check to make sure there are at least two sentences - important when using the sentence mask
        # sentences = punkt_sentences(first_doc)
        # if len(sentences) ==1:
        #     break

        #corpus = normalize_and_remove_stop_words(first_doc)

        # # Store a list of sentences in the cluster at each iteration
        # sentences = []
        # sentences += (data_gen.get_first_and_last_sentence(first_doc))
        task = {"C": "","Q": "", "A": ""}
        for index in sortedEntries[1:]:
            # Find next document in order
            raw_doc = documentData[index]["body_text"]

            #normalize and remove stop words from doc
            doc = lil_spacy.join(word_punct_tokens(xml_normalize(raw_doc)))
            #doc = normalize_and_remove_stop_words(raw_doc)

            # #check to make sure there are at least two sentences - important when using the sentence mask
            # sentences = punkt_sentences(raw_doc)
            # if len(sentences) ==1:
            #     break

            if documentData[index]["novelty"]:
                novelty=True
            else:
                novelty=False

            task["C"] += corpus
            task["Q"] = doc
            task["A"] = novelty
            tasks.append(task.copy())
            corpus+=doc

    return tasks

#TODO this part isn't really working right now....but it might in the future. Clean or delete
def analyze_clusters2(all_clusters, lookup_order, documentData, vector_type, word2vec, word_vector_size, param, in_dict={}):
    #This is mostly cut and paste from data_gen but with some differences
    #TODO in the future fold this into data_gen more....but would need somewhat extensive changes there

    # Prepare to store results of feature assessments
    tasks = []

    #initialize vocab and ivocab to empty dictionaries
    if 'vocab' in in_dict:
        print("using a vocab")
        vocab = in_dict['vocab']
    else:
        vocab = {}
    if 'ivocab' in in_dict:
        ivocab = in_dict['ivocab']
    else:
        ivocab = {}
    if 'word2vec' in in_dict:
        built_word2vec = in_dict['word2vec']
    else:
        built_word2vec = word2vec.copy()

    inputs = []
    answers = []
    input_masks = []
    questions = []

    #Iterate through clusters found in JSON file, do feature assessments,
    #build a rolling corpus from ordered documents for each cluster
    for cluster in all_clusters:
        # Determine arrival order in this cluster
        sortedEntries = [x[1] for x in sorted(lookup_order[cluster], key=lambda x: x[0])]

        first_doc = documentData[sortedEntries[0]]["body_text"]

        # Set corpus to first doc in this cluster and prepare to update corpus with new document vocabulary
        corpus = xml_normalize(first_doc)
        built_word2vec, vocab, ivocab = build_vocab(first_doc, built_word2vec, vocab, ivocab, word_vector_size)

        # Store a list of sentences in the cluster at each iteration
        sentences = []
        sentences += (data_gen.get_first_and_last_sentence(first_doc))
        task = {"C": "","Q": "", "A": ""}
        for index in sortedEntries[1:]:
            # Find next document in order
            raw_doc = documentData[index]["body_text"]

            #normalize and remove stop words from doc
            doc = xml_normalize(raw_doc)
            built_word2vec, vocab, ivocab = build_vocab(doc, built_word2vec, vocab, ivocab, word_vector_size)

            if documentData[index]["novelty"]:
                novelty=True
                answers.append(1)
            else:
                novelty=False
                answers.append(0)

            inp_vector = [process_word(word = w,
                                        word2vec = built_word2vec,
                                        vocab = vocab,
                                        ivocab = ivocab,
                                        to_return = vector_type, silent=True) for w in corpus]

            question_rep = [process_word(word = w,
                                        word2vec = built_word2vec,
                                        vocab = vocab,
                                        ivocab = ivocab,
                                        to_return = vector_type, silent=True) for w in corpus]

            # task["C"] += corpus
            # task["Q"] = doc
            # task["A"] = novelty
            # tasks.append(task.copy())
            # corpus+=doc
            inputs.append(np.vstack(inp_vector).astype(floatX))
            questions.append(np.vstack(question_rep).astype(floatX))
            input_masks.append(np.array([index for index, w in enumerate(doc)], dtype=np.int32))
    print(len(inputs), len(questions), len(answers), len(input_masks))

    results = {}
    results['vocab'] = vocab
    results['ivocab'] = ivocab
    results['word2vec'] = built_word2vec
    results[param+'_inputs'] = inputs
    results[param+'_questions'] = questions
    results[param+'_answers'] = answers
    results[param+'_input_masks'] = input_masks

    return results

def build_vocab(doc, word2vec, vocab, ivocab, word_vector_size, silent=True):

    for word in doc:
        if not word in word2vec:
            create_vector(word, word2vec, word_vector_size, silent)
        if not word in vocab:
            next_index = len(vocab)
            vocab[word] = next_index
            ivocab[next_index] = word

    return word2vec, vocab, ivocab
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
    return list(vector)

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
    elif to_return == "bool":
        if word==True:
            return 1
        else:
            return 0
    elif to_return == "onehot":
        raise Exception("to_return = 'onehot' is not implemented yet")


def process_sent(doc, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=False, encoder_decoder=None, vocab_dict={}):
    document_vector = []

    if to_return=="word2vec":
        document_vector = [process_word(w, word2vec, vocab, ivocab , word_vector_size, to_return, silent=True) for w in doc]
    elif to_return=="skip_thought":
        sentences = punkt_sentences(doc)
        norm_sentences = [normalize.xml_normalize(s) for s in sentences]
        document_vector = [ sk.encode(encoder_decoder, norm_sentences)]
    elif to_return=="one_hot":
        data_gen.run_onehot(doc, vocab_dict)

    return document_vector
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