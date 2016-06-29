import py.test
from src.utils import normalize
import subprocess
import os

'''
TEST FROM PYTHIA
'''
def test_empty_string():
    assert normalize.text_to_words("") == ""

def test_links():
    assert normalize.text_to_words("my link is <http://link.com>") == "link"
    
def test_HTML():
    assert normalize.text_to_words("<p> <title=cats> cats pounce </p>") == "cats pounce"
    
def test_letters():
    assert normalize.text_to_words("19 cats&dogs don't eat?") == "cats dogs don t eat"

def test_lower_case():
    assert normalize.text_to_words("Hi BillyBob JOE") == "hi billybob joe"

def test_stop_words():
    assert normalize.text_to_words("the cat has name") == "cat"
    
def test_combo():
    assert normalize.text_to_words("<p> <title=cats> <body> Cats pounce all the time! <http://catlink.com> is a video of cats JUMPING 10 times!! cool, right? </body></p>") == "cats pounce time video cats jumping times cool right"
