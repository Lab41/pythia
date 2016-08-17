# This code is broken.

import sys

sys.path.append('/Users/chrisn/mad-science/pythia/src/featurizers/')
from training import tools
from gensim.models import Word2Vec
import numpy
import warnings
warnings.filterwarnings('ignore')

training_data_location = '/Users/chrisn/testing/training.txt'
encodings_location = '/Users/chrisn/testing/encodings.npz'
path_to_word2vec = '/Users/chrisn/mad-science/pythia/data/stackexchange/model/word2vecAnime.bin'

holdout_percent = 0.5
sentences = [x.strip() for x in open(training_data_location).readlines()]

embed_map = Word2Vec.load_word2vec_format(path_to_word2vec, binary=True)
model = tools.load_model(embed_map)
# I cannot get this next line to not crash.
#######
encodings = tools.encode(model, sentences)
######
numpy.savez(encodings_location,encodings=encodings)
