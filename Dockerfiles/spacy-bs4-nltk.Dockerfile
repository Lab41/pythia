FROM jupyter/scipy-notebook
MAINTAINER abethke <abethke@iqt.org>

RUN conda install --quiet --yes\
	beautifulsoup4 \
	nltk \
	spacy
RUN python -m spacy.en.download

RUN conda install --quiet --yes -p $CONDA_DIR/envs/python2 python=2.7\
	beautifulsoup4 \
	nltk \
	spacy

RUN bash -c '. activate python2 && \
	python -m spacy.en.download'
