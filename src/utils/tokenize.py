""" Wrap some tokenization capabilities in one place.

Patrick Callier
July 2016
"""

from nltk.tokenize import punkt, WordPunctTokenizer
punkter = punkt.PunktSentenceTokenizer()
word_punct_tokenizer = WordPunctTokenizer()

def punkt_sentences(text):
    return punkter.tokenize(text)

def punkt_sentence_span(text):
    return punkter.span_tokenize(text)

def word_punct_tokens(text):
    return word_punct_tokenizer.tokenize(text)
