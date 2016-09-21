FROM ubuntu:14.04
MAINTAINER Pythia

# Install Anaconda
# From: https://github.com/ContinuumIO/docker-images/tree/master/anaconda
#
# Except where noted below, docker-anaconda is released under the following terms:
#
# (c) 2012 Continuum Analytics, Inc. / http://continuum.io
# All Rights Reserved
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Continuum Analytics, Inc. nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL CONTINUUM ANALYTICS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

RUN apt-get update -q --fix-missing && apt-get install -q -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda2-4.0.0-Linux-x86_64.sh && \
    /bin/bash /Anaconda2-4.0.0-Linux-x86_64.sh -b -p /opt/conda && \
    rm /Anaconda2-4.0.0-Linux-x86_64.sh

RUN apt-get install -q -y curl grep sed dpkg && \
    TINI_VERSION=0.9.0 && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# Set up build tools
RUN apt-get update -q && apt-get install -q -y --no-install-recommends --force-yes \
	build-essential && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT [ "tini", "--" ]

# Add Pythia repo to image and create environments
ADD . /pythia
WORKDIR /pythia
RUN envs/make_envs.sh

# Set py3-pythia as default in container
ENV PYTHONPATH /opt/conda/envs/py3-pythia/lib/python3.5:/pythia:$PYTHONPATH
ENV PATH /opt/conda/envs/py3-pythia/bin:$PATH
ENV CONDA_DEFAULT_ENV py3-pythia
ENV CONDA_ENV_PATH /opt/conda/envs/py3-pythia

# run tests
RUN /bin/bash -c 'export THEANO_FLAGS=device=cpu; export PYTHIA_MONGO_DB_URI=localhost:27017; source activate py3-pythia; pip install pytest-cov; py.test -v --cov=src --cov-report term-missing'

CMD [ "/bin/bash" ]
