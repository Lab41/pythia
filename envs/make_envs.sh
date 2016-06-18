#!/bin/bash
# Create conda environments and Jupyter kernels for Pythia
# Adds Pythia repo root to PYTHONPATH so its source tree can be imported 
# in Python
# Requires Anaconda and Jupyter

if [ "$PYTHIA_ROOT" = "" ]; then
    printf "PYTHIA_ROOT must be defined.\nSuggested usage (will clone Pythia in pwd): PYTHIA_ROOT=pythia make_envs.sh\n"
    exit 1
else
  PYTHIA_ROOT=$(cd "$PYTHIA_ROOT" && pwd)
fi

# Create pythia in home directory if it doesn't exist
if [ ! -d "$PYTHIA_ROOT" ]; then
    echo "Cloning pythia into $PYTHIA_ROOT, adding to PYTHONPATH"
    sleep 3
    git clone https://github.com/Lab41/pythia
else
    echo "Using $PYTHIA_ROOT on PYTHONPATH"
    sleep 3
fi


make_env () {
    env_name=$1
    display_name=$2
    python_version=$3

    # Create conda environments with needful packages readied
    sudo apt-get install -yq libjpeg-dev
    
    search_for_environment="$(conda info -e 2>/dev/null | grep -Po '^ *'$env_name'(?= )' | head -n1)"
    echo "Matched environment line: $search_for_environment"
    source deactivate
    if [ "$search_for_environment" = "$env_name" ]; then
        echo "Environment exists, installing original configuration..."
        source activate $env_name && conda install -y python=$python_version scikit-learn \
            beautifulsoup4 lxml jupyter pandas nltk seaborn gensim
    else
        echo "Creating new environment..."
        conda create -y --name $env_name python=$python_version scikit-learn beautifulsoup4 lxml \
            jupyter pandas nltk seaborn gensim
        # Activate environment
        source activate $env_name
    fi

    
    
    # install tensorflow (CPU) and tflearn (py3.4 only)
    if [ "$python_version" = "3.4" ]; then
        pip install --upgrade \
        https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl && \
        pip install --upgrade tflearn
    fi

    # install theano and keras
    pip install --upgrade nose-parameterized Theano keras

    # install bleeding-edge pylzma (for Stack Exchange)
    pip install git+https://github.com/fancycode/pylzma

    # Install Sacred
    pip install sacred

    # install Jupyter kernel, preserving PYTHONPATH and adding Pythia
    pip install ipykernel

    path_info=$(python -m ipykernel install --user --name $env_name --display-name "$display_name")
    # Now add environment information on the second line of kernel.json
    kernel_path=$(python -c "import re; print(re.sub(r'^.*?(/[^ ]+"$env_name").*$', r'\\1', '$path_info'))")
    sed -i '2i  "env" : { "PYTHONPATH" : "'"$PYTHONPATH:$PYTHIA_ROOT"'" },' "$kernel_path/kernel.json"
    echo "Editing $kernel_path/kernel.json..." && cat "$kernel_path/kernel.json" && echo ""

}

make_env "py3-pythia-tf" "Python 3.4 (Pythia, TF)" "3.4"
make_env "py3-pythia" "Python 3.5 (Pythia/Spark-compatible)" "3.5"
