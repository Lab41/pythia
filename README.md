# Pythia

<img src="assets/pythia_logo.png" width="400" alt="pythia logo" />

## Novelty detection in text corpora

Pythia is Lab41's exploration of approaches to novel content detection. We are interested in making it easier to tell when a document coming into a corpus has something new to say.
We welcome your contributions (please see contributor guidelines in CONTRIBUTING) and attention.

The first demos of Pythia at work can be found in [src/pipelines](src/pipelines). We will work hard to keep these usable and up-to-date as we move forward.

## Example

If you are in the root of the repository and bs4, lxml, and scikit-learn (sklearn) are all installed:

```sh
PYTHONPATH=`pwd`:$PYTHONPATH python src/pipelines/bag_words_log_reg.py
```

If not, you can use our docker image, again sitting in the root of the repository:

```sh
docker run -it -e PYTHONPATH=pythia -v `pwd`:/home/jovyan/work/pythia pcallier/pythia sh -c 'conda install -y lxml && python pythia/src/pipelines/bag_words_log_reg.py'
```
