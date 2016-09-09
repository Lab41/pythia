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
Combining the vectors for query and background can be done by concatenating
them, subtracting one from the other, or other operations.

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

### ST_APPEND (False)
### ST_DIFFERENCE (False)
### ST_PRODUCT (False)
### ST_COS (False)

## Latent Dirichlet Allocation
### LDA_APPEND (False)
### LDA_DIFFERENCE (False)
### LDA_PRODUCT (False)
### LDA_COS (False)
### LDA_TOPICS (50)

## Dynamic Memory Networks
### MEM_NET (False)
### MEM_VOCAB (50)
### MEM_TYPE ('dmn_basic')
### MEM_BATCH (1)
### MEM_EPOCHS (5)
### MEM_MASK_MODE ('word')
### MEM_EMBED_MODE ('word2vec')
### MEM_ONEHOT_MIN_LEN (140)
### MEM_ONEHOT_MAX_LEN (1000)

#word2vec
### W2V_APPEND (False)
### W2V_DIFFERENCE (False)
### W2V_PRODUCT (False)
### W2V_COS (False)
### W2V_PRETRAINED (False)
### W2V_MIN_COUNT (5)
### W2V_WINDOW (5)
### W2V_SIZE (100)
### W2V_WORKERS (3)

#one-hot CNN layer
#The one-hot CNN will use the full_vocab parameters
### CNN_APPEND (False)
### CNN_DIFFERENCE (False)
### CNN_PRODUCT (False)
### CNN_COS (False)

# wordonehot (will not play nicely with other featurization methods b/c not
### WORDONEHOT (False)
### WORDONEHOT_VOCAB (5000)

# Classification Algorithms
## Logistic Regression
### LOG_REG (False)
### LOG_PENALTY ('l2')
### LOG_TOL (1e-4)
### LOG_C (1e-4)

## Support Vector Machine
### SVM (False)
### SVM_C (2000)
### SVM_KERNEL ('linear')
### SVM_GAMMA ('auto')

# XGBoost
### XGB (False)
### XGB_LEARNRATE (0.1)
### XGB_MAXDEPTH (3)
### XGB_MINCHILDWEIGHT (1)
### XGB_COLSAMPLEBYTREE (1)

# Other run parameters
## Resampling
### RESAMPLING (True)
Resample observations by label to achieve a 1-to-1 ratio of positive to 
negative observations
### OVERSAMPLING (False)
Resample so that total number of observations per class is equal to the largest class.
Implies REPLACEMENT=True
### REPLACEMENT (False)
When doin resampling, choose observations with replacement from original samples.

## Save training data for grid search
### SAVEEXPERIMENTDATA (False)
### EXPERIMENTDATAFILE ('data/experimentdatafile.pkl')

## Vocabulary
### VOCAB_SIZE (10000)
### STEM (False)
### FULL_VOCAB_SIZE (1000)
### FULL_VOCAB_TYPE ('character')
### FULL_CHAR_VOCAB ("abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/|_@#$%^&*~`+-=<>()[]{}")

### SEED (None)

### USE_CACHE (False)
