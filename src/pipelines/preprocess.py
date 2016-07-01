from src.utils.normalize import text_to_words
from src.featurizers import skipthoughts

def gen_vocab(features, corpusDict):
    # This conditional needs to be made compatible with the cosine and bog options
    result = [None, None]
    if features.skipthoughts:
        result[1] = skipthoughts.load_model()
    if features.cosine or features.bog:
        # needs preferred vocabulary size passed in (ex: vocabsize=500)
        # needs 'from src.utils.normalize import text_to_words'
        # vocabdict contains the most frequently occurring words in the corpus from #1 to n, with n going as far as vocabsize if possible
        # we should be able to use vocabulary=vocabdict when setting up the CountVectorizer for clusters and new docs
        print("making vocabulary...")
        vocabsize = 500
        index = 0
        vocabdict = dict()
        for word in corpusDict:
            if len(vocabdict) < vocabsize:
                cleantext = text_to_words(word)
                if cleantext != '':
                    if not cleantext in vocabdict:
                        vocabdict[cleantext] = index
                        index+=1
            else: break
        result[0] = vocabdict

    return result

def main(argv):
    features, corpusDict = argv[0], argv[1]
    result = gen_vocab(features, corpusDict)
    vocab, encoder_decoder = result[0], result[1]
    return vocab, encoder_decoder
