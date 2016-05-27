FROM jupyter/scipy-notebook
MAINTAINER abethke <abethke@iqt.org>

# Python 3 installs
RUN conda install --quiet --yes\
	beautifulsoup4 \
	nltk \
	spacy
RUN python -m spacy.en.download

# Python 2 installs
RUN conda install --quiet --yes -p $CONDA_DIR/envs/python2 python=2.7\
	beautifulsoup4 \
	nltk \
	spacy

RUN bash -c '. activate python2 && \
	python -m spacy.en.download'
