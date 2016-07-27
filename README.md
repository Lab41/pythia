# Pythia

<img src="assets/pythia_logo.png" width="400" alt="pythia logo" />

[![CircleCI](https://circleci.com/gh/Lab41/pythia.svg?style=shield)](https://circleci.com/gh/Lab41/pythia)
[![codecov](https://codecov.io/gh/Lab41/pythia/branch/master/graph/badge.svg)](https://codecov.io/gh/Lab41/pythia)


## Novelty detection in text corpora

Pythia is Lab41's exploration of approaches to novel content detection. We are interested in making it easier to tell when a document coming into a corpus has something new to say.
We welcome your contributions (see our [contributor guidelines](CONTRIBUTING.md)) and attention.

## Tests

```sh
docker build -t lab41/pythia .     # runs tests and builds project image
```

## Prerequisites

Our code is written in Python 3. [envs/make_envs.sh](envs/make_envs.sh) will install the necessary dependencies on a Debian/Ubuntu system with Anaconda installed.
