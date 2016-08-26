# Dynamic memory networks in Theano modification
The aim of this repository is to slightly modify the implementation of Dynamic memory networks 
implemented by [YerevaNN](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano) and as described in the [paper by Kumar et al.](http://arxiv.org/abs/1506.07285)
to better apply this methodology to document novelty detection. This also only contains a modification of the dmn basic algoithm but this modification can be replicated for the other methodologies implemented in the Dynamic-memory-networks-in-Theano repository.

## Repository contents

| file | description |
| --- | --- |
| `main.py` | the main entry point to train and test available network architectures on document novelty tasks |
| `dmn_basic.py` | our baseline implementation. It is as close to the original in the paper, except the number of steps in the main memory GRU is fixed. Attention module uses `T.abs_` function as a distance between two vectors which causes gradients to become `NaN` randomly. |
| `utils.py` | tools for working with data tasks and GloVe vectors |
| `nn_utils.py` | helper functions on top of Theano and Lasagne |
| `fetch_glove_data.sh` | shell script to fetch GloVe vectors (by [5vision](https://github.com/5vision/kaggle_allen)) |

## Usage

This implementation is based on Theano and Lasagne. One way to install them is:

    pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
    pip install https://github.com/Lasagne/Lasagne/archive/master.zip

The following bash script will download GloVe vectors.

    ./fetch_glove_data.sh

Use `main.py` to train a network:

    python main.py --network dmn_basic --input_train FOLDER1 --input_test FOLDER2

The states of the network will be saved in `states/` folder. 

## License
The original memory network code was released with the following license:
[The MIT License (MIT)](./LICENSE)
Copyright (c) 2016 YerevaNN