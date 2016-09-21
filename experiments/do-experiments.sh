#!/bin/bash

if test "$#" -ne 2
then
    echo "Usage: do-experiments.sh CONFIG_DIR MONGODB_PATH"
    echo "    CONFIG_DIR - directory of JSON experiment configurations"
    echo "    MONGO_DB_PATH - hostname:port:dbname, e.g. localhost:27017:pythia"
    exit 1
fi

find $1 -iname '*.json' | while read config_file
do
    echo "Experiment config: $config_file"
    experiments/experiments.py -m "$2" with "$config_file"
done
