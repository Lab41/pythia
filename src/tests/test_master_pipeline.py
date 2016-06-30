''' 
tests master pipeline module

system tests checking correct arguments. tested from pythia directory.
'''

import pytest
from src.pipelines import master_pipeline
import subprocess
import os

def test_no_directory():
    '''tests that pipeline quits with the correct exit code if no directory is included'''
    result = subprocess.run(['src/pipelines/master_pipeline.py'])
    assert result.returncode == 2

def test_no_features():
    '''tests that pipeline quits with the correct exit code if no features are included'''
    result = subprocess.run(['src/pipelines/master_pipeline.py', 'dir'])
    assert result.returncode == 1

def test_no_algorithms():
    '''tests that pipeline quits with the correct exit code if no algorithms are included'''
    result = subprocess.run(['src/pipelines/master_pipeline.py', 'dir', '-c'])
    assert result.returncode == 3

def test_successful_run():
    '''tests that pipeline has an exit code of zero (runs all the way through) when the right args are provided'''
    result = subprocess.run(['src/pipelines/master_pipeline.py', "src/data/stackexchange/stack_exchange_data/corpus/anime", "-c", "-s"])
    assert result.returncode == 0