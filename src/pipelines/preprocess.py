import sys
from src.utils.normalize import normalize_and_remove_stop_words

def gen_vocab(features, corpusDict):
    if features.cosine or features.bog:
        # needs preferred vocabulary size passed in (ex: vocabsize=500)
        # needs 'src.utils.normalize.normalize_and_remove_stop_words'
        # vocabdict contains the most frequently occurring words in the corpus from #1 to n, with n going as far as vocabsize if possible
        # we should be able to use vocabulary=vocabdict when setting up the CountVectorizer for clusters and new docs
        print("making vocabulary...",file=sys.stderr)
        vocabsize = 500
        index = 0
        vocabdict = dict()
        for word in corpusDict:
            if len(vocabdict) < vocabsize:
                cleantext = normalize_and_remove_stop_words(word)
                if cleantext != '':
                    if not cleantext in vocabdict:
                        vocabdict[cleantext] = index
                        index+=1
            else: break
        return vocabdict

    return None

def main(argv):
    features, corpusDict = argv[0], argv[1]
    vocab = gen_vocab(features, corpusDict)
    return vocab
