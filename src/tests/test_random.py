import numpy as np
from src.pipelines.master_pipeline import get_args, main as pipeline_main
from src.utils.sampling  import label_sample

def test_random():
    args = get_args(SEED=41, 
                    directory='data/stackexchange/anime', 
                    XGB=True,
                    LDA_APPEND=True,
                    BOW_APPEND=True,
                    RESAMPLING=False)
    run1 = pipeline_main(args)
    run2 = pipeline_main(args)
    assert run1==run2
        
def test_random_logreg():
    args = get_args(SEED=41, 
                    directory='data/stackexchange/anime', 
                    XGB=False,
                    LOG_REG=True,
                    LDA_APPEND=False,
                    BOW_APPEND=True,
                    W2V_APPEND=False,
                    RESAMPLING=False)
    run1 = pipeline_main(args)
    run2 = pipeline_main(args)
    assert run1==run2

def test_random_xgboost():
    args = get_args(SEED=41,
                    directory='data/stackexchange/anime',
                    XGB=True,
                    LDA_APPEND=False,
                    BOW_APPEND=True,
                    W2V_APPEND=False,
                    RESAMPLING=False)
    run1 = pipeline_main(args)
    run2 = pipeline_main(args)
    assert run1==run2

def test_random_svm():
    args = get_args(SEED=41, 
                    directory='data/stackexchange/anime', 
                    XGB=False,
                    SVM=True,
                    LDA_APPEND=False,
                    BOW_APPEND=True,
                    W2V_APPEND=False,
                    RESAMPLING=False)
    run1 = pipeline_main(args)
    run2 = pipeline_main(args)
    assert run1 == run2

def test_random_lda():
    args = get_args(SEED=41, 
                    directory='data/stackexchange/anime', 
                    XGB=False,
                    LOG_REG=True,
                    LDA_APPEND=True,
                    BOW_APPEND=False,
                    W2V_APPEND=False,
                    RESAMPLING=False)
    run1 = pipeline_main(args)
    run2 = pipeline_main(args)
    assert run1 == run2

def test_random_resampling():
    data = [ { 'key': True, 'data': 123 },
             { 'key': True, 'data': 123 },
             { 'key': True, 'data': 123 },
             { 'key': True, 'data': 123 },
             { 'key': True, 'data': 123 },
             { 'key': False, 'data': 123 },
             { 'key': False, 'data': 123 },
            ]
    def get_state(seed):
        return np.random.RandomState(seed)
    sampled_data = label_sample(data, 'key', random_state=get_state(41))
    sampled_data2 = label_sample(data, 'key', random_state=get_state(41))
    assert sampled_data == sampled_data2
