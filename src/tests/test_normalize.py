import py.test
from src.utils import normalize
import subprocess
import os

'''
TEST FROM PYTHIA
'''
def test_empty_string():
    assert normalize.normalize_and_remove_stop_words("") == ""

def test_links():
    assert normalize.remove_links("my link is <http://link.com>") == "my link is <http:>"

def test_HTML():
    assert normalize.normalize_and_remove_stop_words("<p> <title=cats> cats pounce </p>") == "cats pounce"

def test_letters():
    assert normalize.normalize_and_remove_stop_words("19 cats&dogs don't eat?") == "cats dogs don t eat"

def test_lower_case():
    assert normalize.normalize_and_remove_stop_words("Hi BillyBob JOE") == "hi billybob joe"

def test_stop_words_text():
    assert normalize.normalize_and_remove_stop_words("the cat has name") == "cat"

def test_stop_words():
    assert normalize.remove_stop_words(["the", "fool", "is", "mine"]) == ["fool"]
    assert normalize.remove_stop_words(["the", "fool", "is", "mine"], ["fool"]) == ['the', 'is', 'mine']

def test_xml_normalize():
    assert normalize.xml_normalize("my link is <http://link.com>. Enough.") == 'my link is . enough.'
    
def test_combo():
    assert normalize.normalize_and_remove_stop_words("<p> <title=cats> <body> Cats pounce all the time! <http://catlink.com> is a video of cats JUMPING 10 times!! cool, right? </body></p>") == "cats pounce time video cats jumping times cool right"
