#!/bin/bash
# Create conda environments and Jupyter kernels for Pythia
# Adds Pythia repo root to PYTHONPATH so its source tree can be imported 
# in Python
# Requires Anaconda and Jupyter

PYTHIA_CONFIG="$1"
if [ "$PYTHIA_CONFIG" = "" ]; then
    printf "Must pass in JSON object of configuration variables.\nSuggested usage:"
    printf "make_envs.sh <(echo \"{PYTHONPATH=\\\\\"\$PYTHONPATH\\\\\"}\")\n"
    exit 1
else

fi


make_env () {
    env_name="$1"
    display_name="$2"
    python_version="$3"
    
    echo $env_name
    echo $display_name
    echo $python_version
    echo $PYTHIA_CONFIG


    search_for_environment="$(conda info -e 2>/dev/null | grep -Po '^ *'$env_name'(?= )' | head -n1)"
    echo "Matched environment line: $search_for_environment"
    source deactivate 2>/dev/null
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

    # Install the kernel and retrieve its destination directory
    path_info=$(python -m ipykernel install --user --name $env_name --display-name "$display_name")
    
    # Now add environment information on the second line of the new env's kernel.json
    kernel_dir=$(python -c "import re; print(re.sub(r'^.*?(/[^ ]+"$env_name").*$', r'\\1', '$path_info'))")
    kernel_path="$kernel_dir/kernel.json"
    echo "Editing $kernel_path..."
    cat <(sed -n '1p' "$kernel_path") \
        <(echo "\"env\" : ") \
        "$PYTHIA_CONFIG" \
        <(echo ", ") \
        <(sed '1d' "$kernel_path" ) > /tmp/kernel.json
    mv /tmp/kernel.json "$kernel_path"
    
    cat "$kernel_path" && echo ""

}

make_env "py3-pythia-tf" "Python 3.4 (Pythia, TF)" "3.4"
make_env "py3-pythia" "Python 3.5 (Pythia/Spark-compatible)" "3.5"
