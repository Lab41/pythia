#!/bin/bash
# Create conda environments and Jupyter kernels for Pythia
# Adds Pythia repo root to PYTHONPATH so its source tree can be imported
# in Python
# Requires Anaconda and Jupyter

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

set -e

PYTHIA_CONFIG="$1"
if [ "$PYTHIA_CONFIG" = "" ]; then
    printf "Did not pass in JSON file of configuration variables.\n"
    printf "Continuing with PYTHONPATH pass-through...\n"
    printf "Alternative usage: \n"
    printf "\tmake_envs.sh config.json\n\n"
    PYTHIA_CONFIG='{ "PYTHONPATH" : "'"$PYTHONPATH"'" }'
else
    PYTHIA_CONFIG="$(cat $PYTHIA_CONFIG)"
fi


make_env () {
    env_name="$1"
    display_name="$2"
    python_version="$3"

    echo $env_name
    echo $display_name
    echo $python_version
    echo $PYTHIA_CONFIG


set +e
    # Does not work with BSD grep (OS X)
    search_for_environment="$(conda info -e 2>/dev/null | grep -Po '^ *'$env_name'(?= )' | head -n1)"
    echo "Matched environment line: $search_for_environment"
    source deactivate 2>/dev/null || true
set -e
    sleep 2
    if [ "$search_for_environment" = "$env_name" ]; then
        echo "Environment exists, installing original configuration..."
        sleep 2
        source activate $env_name && conda install -y python=$python_version scikit-learn \
            beautifulsoup4 lxml jupyter pandas nltk seaborn gensim pip==8.1.1 pymongo
    else
        echo "Creating new environment..."
        sleep 2
        conda create -y --name "$env_name" python="$python_version" scikit-learn beautifulsoup4 lxml \
            jupyter pandas nltk seaborn gensim pip==8.1.1 pymongo
        # Activate environment
        source activate "$env_name"
    fi

    # install tensorflow
    conda install -y -c conda-forge tensorflow

    # Download some NLTK data (punkt tokenizer)
    python -m nltk.downloader punkt

    # Install XGBoost classifier
    pip install xgboost

    # install theano and keras
    pip install nose-parameterized Theano keras

    # install Lasagne
    pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
    pip install https://github.com/Lasagne/Lasagne/archive/master.zip

    # install bleeding-edge pylzma (for Stack Exchange)
    pip install git+https://github.com/fancycode/pylzma

    # Install Sacred (with patch for parse error)
    # pip install sacred
    pip install docopt pymongo
    save_dir=`pwd`
    rm -rf /tmp/sacred || true
    git clone https://github.com/IDSIA/sacred /tmp/sacred
    cd /tmp/sacred
    git checkout 0.6.8
    git apply "$script_dir/requirement_parse_patch.txt"
    python setup.py install
    cd "$save_dir"

    # install Jupyter kernel, preserving PYTHONPATH and adding Pythia
    pip install ipykernel

    # Install the kernel and retrieve its destination directory
    path_info=$(python -m ipykernel install --user --name $env_name --display-name "$display_name")

    # Now add environment information on the second line of the new env's kernel.json
    kernel_dir=$(python -c "import re; print(re.sub(r'^.*?(/[^ ]+"$env_name").*$', r'\\1', '$path_info'))")
    kernel_path="$kernel_dir/kernel.json"
    echo "Editing $kernel_path..."
    cat <(sed -n '1p' "$kernel_path") \
        <(echo "\"env\" : ") \
        <(echo "$PYTHIA_CONFIG") \
        <(echo ", ") \
        <(sed '1d' "$kernel_path" ) > /tmp/kernel.json
    mv /tmp/kernel.json "$kernel_path"

    cat "$kernel_path" && echo ""

}

<<<<<<< HEAD
=======
make_env "py3-pythia-tf" "Python 3.4 (Pythia, TF)" "3.4"
>>>>>>> roll memory network alg into pipeline
make_env "py3-pythia" "Python 3.5 (Pythia/Spark-compatible)" "3.5"
