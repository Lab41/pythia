# Pythia

<img src="assets/pythia_logo.png" width="400" alt="pythia logo" />

[![CircleCI](https://circleci.com/gh/Lab41/pythia.svg?style=svg)](https://circleci.com/gh/Lab41/pythia)
[![codecov](https://codecov.io/gh/Lab41/pythia/branch/master/graph/badge.svg)](https://codecov.io/gh/Lab41/pythia)


## Novelty detection in text corpora

Pythia is Lab41's exploration of approaches to novel content detection. We are interested in making it easier to tell when a document coming into a corpus has something new to say.
We welcome your contributions (see our [contributor guidelines](CONTRIBUTING.md)) and attention.

The first demos of Pythia at work can be found in [src/pipelines](src/pipelines). We will work hard to keep these usable and up-to-date as we move forward.

## Example

If you are in the root of the repository, the following will train and evaluate a bag-of-words logistic regression on the 'Anime' portion of the Stack Exchange corpus:

```sh
PYTHONPATH=`pwd`:$PYTHONPATH python src/pipelines/master_pipeline.py data/stack_exchange/corpus/anime --bag_of_words --log_reg
```

## Prerequisites

Our code is written in Python 3. [envs/make_envs.sh](envs/make_envs.sh) will install the necessary dependencies on a Debian/Ubuntu system with Anaconda installed.
