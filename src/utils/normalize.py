from bs4 import BeautifulSoup
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
import re
import warnings
from nltk import stem


link_re = re.compile(r'\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*')
letter_re = re.compile(r"[^a-zA-Z]")

def normalize_and_remove_stop_words(raw_text, stem=False, **kwargs):
    '''
    Algorithm to convert raw text to a return a clean text string
    Method modified from code available at:
    https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
    Args:
        raw_text: Original text to clean and normalize
    Returns:
        clean_text: Cleaned text, converted to lower case, with punctuation removed
        and one space between each word.
    '''
    # 1. Remove web links
    links_removed = remove_links(raw_text)
    #
    # 2. Remove HTML
    #TODO Potentially look into using package other than BeautifulSoup for this step
    # Suppress UserWarnings from BeautifulSoup due to text with tech info (ex: code, directory structure)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        review_text = BeautifulSoup(links_removed, "lxml").get_text()

    #
    # 3. Remove non-letters
    letters_only = letter_re.sub(" ", review_text)
    #
    # 4. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 5. Remove stop words
    meaningful_words = remove_stop_words(words)


    #6. stem if necessary
    if stem: meaningful_words = porter_stem(meaningful_words)
    #
    # 7. Join the words back into one string separated by space,
    # and return the result.
    clean_text = ( " ".join( meaningful_words ))
    return clean_text

def porter_stem(words):
    stemmer = stem.PorterStemmer()
    return [stemmer.stem(w) for w in words]

def xml_normalize(raw_text, stem=False, **kwargs ):
    """Alternative normalization: HTML/XML and URLs stripped out, lower-cased,
    but stop words and punctuation remain."""
    # 1. Remove web links
    links_removed = remove_links(raw_text)

    # 2. Remove HTML
    #TODO Potentially look into using package other than BeautifulSoup for this step
    # Suppress UserWarnings from BeautifulSoup due to text with tech info (ex: code, directory structure)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        review_text = BeautifulSoup(links_removed, "lxml").get_text()

    # 3. Convert to lower-case
    lower_case = review_text.lower().split()

    # 4. Stemp if necessary
    if stem: lower_case = porter_stem(lower_case)
    #
    # 7. Join the words back into one string separated by space,
    # and return the result.
    clean_text = ( " ".join( lower_case ))

    return clean_text

def remove_links(text):
    links_removed = link_re.sub('', text)
    return links_removed

def remove_stop_words(words, stop_words=ENGLISH_STOP_WORDS):
    """Remove stop words from input"""
    return [w for w in words if not w in stop_words]
