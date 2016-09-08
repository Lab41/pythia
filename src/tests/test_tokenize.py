from src.utils import tokenize

def test_punkt():
    """Test sentence tokenization"""

    assert tokenize.punkt_sentences("S1. S2. S3! S4!!!") == ["S1.", "S2.", "S3!", "S4!!", "!"]
    assert tokenize.punkt_sentences("S1.      S4!!!") == ["S1.", "S4!!", "!"]

def test_word_punct():
    """Test regex-based word and punctuation tokenization"""

    assert tokenize.word_punct_tokens("Who are you??? Stop, now!") == \
        ["Who", "are", "you", "???", "Stop", ",", "now", "!"]
