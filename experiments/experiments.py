#!/usr/bin/env python

import sys
import os
import pprint
import subprocess
import json

from sacred import Experiment
from sacred.observers import MongoObserver

from src.pipelines.master_pipeline import main as pythia_main
from src.pipelines.master_pipeline import parse_args

ex_name='pythia_experiment'
db_name='sacred_demo'

def set_up_xp():
    # Check that MongoDB config is set
    try:
        mongo_uri=os.environ['PYTHIA_MONGO_DB_URI']
    except KeyError as e:
        print("Must define location of MongoDB in PYTHIA_MONGO_DB_URI for observer output",file=sys.stderr)
        raise

    ex = Experiment(ex_name)
    ex.observers.append(MongoObserver.create(url=mongo_uri,
                                         db_name=db_name))
    return ex

def main():
    xp = set_up_xp()
    args=parse_args(["--bag-of-words", "--log-reg", "data/stackexchange/anime"])
    pythia_main(args)
    #xp_results = subprocess.run(["src/pipelines/master_pipeline.py",
    #    "--bag-of-words", "--log-reg", "data/stackexchange/anime"],
    #    stdout=subprocess.PIPE, universal_newlines=True)
    #pprint.pprint(xp_results.stdout, stream=sys.stderr)
    #print(json.loads(xp_results.stdout), file=sys.stderr)

if __name__=="__main__":
    main()
