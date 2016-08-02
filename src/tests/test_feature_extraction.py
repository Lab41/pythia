import py.test
from src.pipelines import data_gen


def test_onehot():
    """Test one-hot document generation"""

    doc = ["hello", "you", "wanton", "civet", ",", "you"]
    vocab = {   "hello" : 0,
                "you"   : 1,
                "civet" : 2,
                ","     : 3
            }

    doc_onehot = data_gen.run_onehot(doc, vocab)
    doc_onehot_minlength = data_gen.run_onehot(doc, vocab, min_length = 10)
    doc_onehot_maxlength = data_gen.run_onehot(doc, vocab, max_length = 2)
    print(doc_onehot)
