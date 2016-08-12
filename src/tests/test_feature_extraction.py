import pytest
import numpy as np
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
    doc_onehot_encoded = np.array(
        [[ 1.,  0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.,  1.],
        [ 0.,  0.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  1.,  0.]], dtype=np.float32)

    # encoding is correct
    assert (doc_onehot == doc_onehot_encoded).all()
    # minimum length correctly enforced
    assert doc_onehot_minlength.shape == (4, 10)
    # maximum length correctly enforced
    assert doc_onehot_maxlength.shape == (4, 2)

if __name__=="__main__":
    test_onehot()
