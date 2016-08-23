import py.test
from src.pipelines.master_pipeline import get_args, main as pipeline_main


#FIXME: Should make so many of these for different arguments
def test_random():
    args = get_args(SEED=41, 
                    directory='data/stackexchange/anime', 
                    XGB=True,
                    LDA_APPEND=True,
                    BOW_APPEND=True,
                    W2V_APPEND=True,
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

def test_random_resampling():
    args = get_args(SEED=41, 
                    directory='data/stackexchange/anime', 
                    XGB=True,
                    BOW_APPEND=True,
                    RESAMPLING=True)
    
    run1 = pipeline_main(args)
    run2 = pipeline_main(args)
    assert run1==run2
