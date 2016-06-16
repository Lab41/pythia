#!/bin/bash

# Create pythia in home directory if it doesn't exist
if [ ! -d "$HOME/pythia" ]; then
    echo "Cloning pythia into $HOME, adding to PYTHONPATH"
    sleep 3
    git clone https://github.com/Lab41/pythia
else
    echo "Using $HOME/pythia on PYTHONPATH"
    sleep 3
fi
PYTHIA_ROOT="$HOME/pythia"

make_env () {
env_name=$1
display_name=$2
python_version=$3

# Create conda environments with needful packages readied
if [ "$(conda info -e 2>/dev/null | grep -o '^ *'$env_name'')" = "$env_name" ]; then
    echo "Environment exists, installing original configuration..."
    source activate $env_name && conda install python=$python_version scikit-learn beautifulsoup4 lxml jupyter
else
    echo "Creating new environment..."
    conda create --name $env_name python=$python_version scikit-learn beautifulsoup4 lxml jupyter
fi

# install tensorflow (CPU) and tflearn (py3.4 only)
if [ "$python_version" = "3.4" ]; then
    source activate $env_name && \
        pip install --upgrade \
            https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl && \
        pip install --upgrade tflearn
fi

# install theano and keras
source activate $env_name && \
    pip install --upgrade nose-parameterized Theano keras

# install bleeding-edge pylzma
source activate $env_name && \
    pip install git+https://github.com/fancycode/pylzma

# install Jupyter kernel, preserving PYTHONPATH and adding Pythia
source activate $env_name && \
    pip install ipykernel && \
    path_info=$(python -m ipykernel install --user --name $env_name --display-name "$display_name") && \
    # Now add environment information on the second line of kernel.json
    kernel_path=$(python -c "import re; print(re.sub(r'^.*?(/[^ ]+"$env_name").*$', r'\\1', '$path_info'))") && \
    sed -i '2i  "env" : { "PYTHONPATH" : "'"$PYTHONPATH":"$PYTHIA_ROOT"'" },' "$kernel_path/kernel.json" && \
    echo "Editing " $kernel_path/kernel.json"..." && cat "$kernel_path/kernel.json" && echo && sleep 3
}

make_env py3-pythia-tf "Python 3.4 (Pythia, TF)" "3.4"
make_env py3-pythia "Python 3.5 (Pythia/Spark)" "3.5"