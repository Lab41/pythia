#!/usr/bin/env python

import subprocess
import sacred

def main():
    subprocess.run(["src/pipelines/master_pipeline.py", "--bag-of-words", "--log-reg", "data/stackexchange/anime"])

if __name__=="__main__":
    main()
