# Pythia

## Detecting novelty and redundancy in text

Pythia is Lab41's exploration of approaches to novel content detection.

# Usage and Runtime Options

`experiments/experiments.py with OPTION=value OPTION_2=value_2 OPTION_3='value with spaces'`

# Terminology

# Usage and runtime options

## Data location
### directory ('data/stackexchange/anime')

# Featurization techniques

## Bag of words

Bag-of-words features can be generated for query and background documents.
Aggregating the vectors for query document and background documents can be done by concatenating
them, subtracting one from the other, or other operations. A temporal $tf-idf$ score is also 
available in this family of settings.

Bag-of-words vectors will automatically be used if any of the following aggregation
parameters is set to `True`:

### BOW_APPEND (False)

Calculate bag-of-words vectors for query document, background documents. Concatenate 
query document and sum of vectors for background documents.  

### BOW_DIFFERENCE (False)

Use difference vector, i.e. `bow(query) - bow(background)`
### BOW_PRODUCT (False)
Take the product of bag-of-words vectors
### BOW_COS (False)
Take the cosine similarity of query and background vectors
### BOW_TFIDF (False)
Take the temporal TF-IDF score for the cluster

## Skip-thought vectors

Skip-thought vectors are a method for representing the structure and content
of sentences in a fixed-size, dense vector. They can be concatenated, subtracted,
multiplied elementwise, or compared with the cosine distance.

Description of method: [https://arxiv.org/abs/1506.06726](https://arxiv.org/abs/1506.06726)  
Basis of implementation: [https://github.com/ryankiros/skip-thoughts](https://github.com/ryankiros/skip-thoughts)

See notes on Bag-of-Words for all of the below options for aggregating skip-thought feature vectors:  
### ST_APPEND (False)
Concatenate vectors
### ST_DIFFERENCE (False)
Difference of vectors
### ST_PRODUCT (False)
Product of vectors
### ST_COS (False)


## Latent Dirichlet Allocation

Latent Dirichlet Allocation (LDA) is a widely-used method for representing the topic 
distribution in a corpus of documents. Given a corpus of $M$ documents with $N$ unique
words, LDA yields two matrices, one $M \times k$ repesenting the 'weight' of each
document across the $k$ possible topics, and one $k \times N$ describing which unique words 
are associated with each topic. 

LDA posits that co-occurring words within documents are
'generated' by the hidden topic variables with document-specific frequencies, so it is
a good way of expressing the assumptions that a) some words naturally co-occur with each other
and b) which co-occurrence patterns are relevant for a given document depend on what
the document is talking about.

Original paper: [http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)  
Library used: [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)

### LDA_TOPICS (50)
The number of possible topics represented in the corpus. Higher is often better as LDA should learn not to use unnecessary topics; in
practice some tuning is usually necessary to find a happy medium.
### LDA_APPEND (False)
### LDA_DIFFERENCE (False)
### LDA_PRODUCT (False)
### LDA_COS (False)

## Dynamic Memory Networks

Dynamic memory networks (DMN) are a 2016 deep learning architecture designed for automatic question answering. They take in
natural language representations of background knowledge and natural language representations of a query and learn
a decoding function to provide an answer. In our adaptation of DMN, background documents are fed in to the background knowledge
module, the query document is used as the query, and the possible responses are True (novel) or False.

Original manuscript: [http://arxiv.org/abs/1506.07285](http://arxiv.org/abs/1506.07285) 
Basis implementation: [https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano)

### MEM_NET (False)
Set to True to use DMN algorithm
### MEM_VOCAB (50)
Vocabulary size for encoding functions. Ideally, this would be rather large, but memory and processing constraints may force 
the use of unnaturally small values.
### MEM_TYPE ('dmn_basic')
Architectural variant to use. `'dmn_basic'` is the only supported value at present.
### MEM_BATCH (1)
Minibatch size for gradient-based training of DMN. Currently values other than 1 are unsupported.
### MEM_EPOCHS (5)
Number of training epochs to conduct.
### MEM_MASK_MODE ('word')
Accepts values `'word'` and `'sentence'`. Tells DMN which unit to treat as an 'epsiode' to 
encode in memory.
### MEM_EMBED_MODE ('word2vec')
### MEM_ONEHOT_MIN_LEN (140)
If `MEM_MASK_MODE` is `'word_onehot'`, set the minimum length of a one-hot-encoded document
### MEM_ONEHOT_MAX_LEN (1000)
Maximum length of a one-hot-encoded document

#word2vec

word2vec is a popular algorithm for learning 'distributed' vector representations of words. In practice,
word2vec learns to represent each unique word in a corpus as a 50- to 300-dimensional vector of real
numbers, providing plenty of room to account for semantic and syntactic similarities and differences between words.

In Pythia, documents are represented by word2vec by finding vectors representing the individual words in 
the input documents and aggregating these vectors in creative ways. Word vectors are extracted from
the first and last sentences of input documents and then combined using averaging, concatenation, 
elementwise max, elementwise min, or absolute elementwise max. Once query and background vectors
have been generated, they are combined using any of the customary aggregation techniques (see bag of words
for discussion)

Original paper: []()
Implementation used: [](gensim)

Aggregating query and background vectors:  
### W2V_APPEND (False)
### W2V_DIFFERENCE (False)
### W2V_PRODUCT (False)
### W2V_COS (False)

Other parameters:  
### W2V_PRETRAINED (False)
Use pretrained model? This should be available in the directory described by the 
PYTHIA_MODELS_PATH environment variable. Currently only the 300-dimensional
Google News model is supported, so this should have the file name `GoogleNews-vectors-negative300.bin` or
`GoogleNews-vectors-negative300.bin.gz`

If `W2V_PRETRAINED` is False, Pythia will train a word2vec model based on your corpus (not recommended for small collections).
 THe following parameters control word2vec training.
### W2V_MIN_COUNT (5)
Minimum number of times a unique word must appear in corpus to be given a vector representation
### W2V_WINDOW (5)
Window size, in words, to the left or right of the word being trained.
### W2V_SIZE (100)
Dimensionality of trained word vectors.
### W2V_WORKERS (3)
If >1, number of cores to do parallel training on. Parallel-trained word2vec models will converge much more quickly 
but training behavior is non-deterministic and not strictly replicable.

# One-hot CNN activation features
The one-hot CNN will use the full_vocab parameters
### CNN_APPEND (False)
### CNN_DIFFERENCE (False)
### CNN_PRODUCT (False)
### CNN_COS (False)

# Word-level one-hot encoding
Not currently used.
### WORDONEHOT (False)
Use word-level one-hot encoding?
### WORDONEHOT_VOCAB (5000)

# Classification Algorithms

If traditional (non-DMN) featurization techniques are chosen, a classifier must also be selected. Pythia
supports batch logistic regression, batch SVM, and the popular boosting algorithm XGBoost, as well as SGD-based
(minibatch) logistic regression and linear SVM, which may have favorable memory performance for very large corpora.

## Logistic Regression
Tried-and-true statistical classification technique. Learns a linear combination of input features and
applies a nonlinear transform to output a hypothesis between 0.0 and 1.0, with values equal to or above 
0.5 typically taken as true and the rest as false.

Pythia uses [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) to do logistic regression.

### LOG_REG (False)
Set to True to use logistic regression
### LOG_PENALTY ('l2')
Form of regularization penalty to use (see scikit-learn docs)
### LOG_TOL (1e-4)
Convergence criterion during model fitting
### LOG_C (1e-4)
Inverse regularization strength

## Support Vector Machine
Nonparametric classifer. Can take a prohibitively long time to converge for large datasets.

Also uses [scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)'s implementation.
### SVM (False)
Set to True to use SVM
### SVM_C (2000)
Inverse regularization strength
### SVM_KERNEL ('linear')
Kernel function to use. Can be any predefined setting accepted by `sklearn.svm.SVC`
### SVM_GAMMA ('auto')
If kernel is `'poly'`, `'rbf'`, or `'sigmoid'`, the kernel coefficient.

# XGBoost

Boosted decision tree algorithm. Fast and performant, but may not scale to much larger datasets.

Original manuscript: [http://arxiv.org/abs/1603.02754](http://arxiv.org/abs/1603.02754)
Implementation used: [https://github.com/dmlc/xgboost](https://github.com/dmlc/xgboost)

### XGB (False)
Set to true to use XGBoost
### XGB_LEARNRATE (0.1)
"Learning rate" (see documentation)
### XGB_MAXDEPTH (3)
Maximum depth of tree
### XGB_MINCHILDWEIGHT (1)
Typically, minimum number of children allowed in any child node
### XGB_COLSAMPLEBYTREE (1)
Proportion (0 to 1) of features to sample when building a tree

# Other run parameters
## Resampling
### RESAMPLING (True)
Resample observations by label to achieve a 1-to-1 ratio of positive to negative observations
### OVERSAMPLING (False)
Resample so that total number of observations per class is equal to the largest class. 
Implies REPLACEMENT=True
### REPLACEMENT (False)
When doing resampling, choose observations with replacement from original samples.

## Save training data for grid search
When using `experiments/conduct_grid_search.py`, use these variables
to allow GridSearchCV to cooperate with the Pythia pipeline.
### SAVEEXPERIMENTDATA (False)
### EXPERIMENTDATAFILE ('data/experimentdatafile.pkl')

## Vocabulary

Two seaparate vocabularies are computed. One, a reduced vocabulary, excludes stop words and punctuation tokens, and the
other, `FULL_VOCAB`, retains them. `FULL_VOCAB` can also be set to use character-level tokenization 
instead of word-level tokenization.

### VOCAB_SIZE (10000)
Size of reduced vocabulary
### STEM (False)
Conduct stemming?
### FULL_VOCAB_SIZE (1000)
Number of unique tokens in word-level full vocabulary.
### FULL_VOCAB_TYPE ('character')
Either `'word'` or `'character'`. Determines tokenization strategy for full vocabulary
### FULL_CHAR_VOCAB ("abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/|_@#$%^&*~`+-=<>()[]{}")

### SEED (41)
Random number generator seed value.

### USE_CACHE (False)
Cache preprocessed JSON documents in `./.cache`; this can reduce experiment time significantly for large corpora.
