#!/bin/bash
# Create conda environments and Jupyter kernels for Pythia
# Adds Pythia repo root to PYTHONPATH so its source tree can be imported
# in Python
# Requires Anaconda and Jupyter

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

set -e

PYTHIA_CONFIG="$1"
if [ "$PYTHIA_CONFIG" = "" ]; then
    jsonfile=0
    printf "Did not pass in JSON file of configuration variables.\n"
    printf "Continuing with PYTHONPATH pass-through...\n"
    printf "Alternative usage: \n"
    printf "\tmake_envs.sh config.json\n\n"
    PYTHIA_CONFIG='{ "PYTHONPATH" : "'"$PYTHONPATH"'" }'
else
    jsonfile=1
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
    if [ ! "$search_for_environment" = "$env_name" ]; then
        echo "Creating new environment..."
        sleep 2
        conda create -q -y --name "$env_name" python="$python_version"
    fi

    # basics
    source activate "$env_name"

    # If a JSON config file was provided, map Pythia env variables whenever Pythia conda env is activated
    if [ $jsonfile != 0 ]; then
        # Look for Pythia's Anaconda environment path in system environment variables, depending on conda version
        envloc=0
        if [ ! -z "$CONDA_PREFIX" ]; then
            envloc=$CONDA_PREFIX
        else
            if [ ! -z "$CONDA_ENV_PATH" ]; then
                envloc=$CONDA_ENV_PATH
            fi
        fi

        # If Anaconda's env path was found, map Pythia env variables from JSON config file to Pythia conda env
        if [ $envloc != 0 ]; then
            mkdir -p $envloc/etc/conda/activate.d
            mkdir -p $envloc/etc/conda/deactivate.d
            touch $envloc/etc/conda/activate.d/env_vars.sh
            touch $envloc/etc/conda/deactivate.d/env_vars.sh

            # Search PYTHIA_CONFIG's JSON format for the PYTHIA_MODELS_PATH property and return PYTHIA_MODELS_PATH's value
            modelval=$( (echo $PYTHIA_CONFIG) | (awk -F"[,:}]" '{for(i=1;i<=NF;i++){if($i~/'PYTHIA_MODELS_PATH'\042/){print $(i+1)}}}' | tr -d '"' | tr -d '[[:space:]]') )
            if [ "$modelval" != "" ]; then
                echo "export PYTHIA_MODELS_PATH=$modelval" >> $envloc/etc/conda/activate.d/env_vars.sh
                echo "unset PYTHIA_MODELS_PATH" >> $envloc/etc/conda/deactivate.d/env_vars.sh
            fi
        fi
    fi

    conda install -q -y python="$python_version" scikit-learn \
        beautifulsoup4==4.4.1 lxml==3.6.1 jupyter==1.0.0 pandas==0.18.1 nltk==3.2.1 \
        seaborn==0.7.1 gensim==0.12.4 pip==8.1.1 pymongo==3.0.3 pytest \
        h5py psutil memory_profiler

    # install tensorflow
    conda install -q -y -c conda-forge tensorflow==0.9.0

    # Download some NLTK data (punkt tokenizer)
    python -m nltk.downloader punkt

    # Install XGBoost classifier
    pip install -q xgboost==0.4a30

    # Install auto-sklearn (Linux only) and dependencies for experimentation and hyperparameter optimization
    # http://automl.github.io/auto-sklearn/stable/index.html
    pip install -q Cython==0.24.1
    pip install -q -r https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt
    pip install auto-sklearn==0.0.1

    # install theano and keras
    pip install -q nose-parameterized==0.5.0 Theano==0.8.2 keras==1.0.7

    # install Lasagne
    pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
    pip install https://github.com/Lasagne/Lasagne/archive/master.zip

    # install bleeding-edge pylzma (for Stack Exchange)
    pip install -q git+https://github.com/fancycode/pylzma@996570e

    # Install Sacred (with patch for parse error)
    # pip install sacred
    pip install -q docopt==0.6.2 pymongo==3.0.3
    save_dir=`pwd`
    rm -rf /tmp/sacred || true
    git clone https://github.com/IDSIA/sacred /tmp/sacred
    cd /tmp/sacred
    git checkout 0.6.8
    git apply "$script_dir/requirement_parse_patch.txt"
    python setup.py install
    cd "$save_dir"

    # Install Hyperopt
    pip install git+https://github.com/Lab41/hyperopt.git

    # install Jupyter kernel, preserving PYTHONPATH and adding Pythia
    pip install -q ipykernel==4.3.1

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
    echo "Finished configuring kernel."
}

make_env "py3-pythia" "Python 3.5 (Pythia)" "3.5"
