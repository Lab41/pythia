''' 
tests master pipeline module

system tests checking correct arguments. tested from pythia directory.
'''

import pytest
from src.pipelines import master_pipeline
import subprocess
import os

# def setup_module(module):
#     '''changes current working directory to the pipeline'''
#     os.chdir('src/pipelines')
    
# def teardown_module(module):
#     '''changes current working directory back to pythia'''
#     os.chdir('../..')

def test_no_directory():
    '''tests that pipeline quits with the correct exit code if no directory is included'''
    # dir_ = os.getcwd()
    # file_ = os.path.join(dir_, "src/pipelines/master_pipeline.py")
    result = subprocess.run(['src/pipelines/master_pipeline.py'])
    assert result.returncode == 2

def test_no_features():
    '''tests that pipeline quits with the correct exit code if no features are included'''
    dir_ = os.getcwd()
    file_ = os.path.join(dir_, "master_pipeline.py")
    result = subprocess.run([file_, "dir"])
    assert result.returncode == 1

def test_no_algorithms():
    '''tests that pipeline quits with the correct exit code if no algorithms are included'''
    dir_ = os.getcwd()
    file_ = os.path.join(dir_, "master_pipeline.py")
    result = subprocess.run([file_, "dir", "-c"])
    assert result.returncode == 3

def test_successful_run():
    '''tests that pipeline has an exit code of zero (runs all the way through) when the right args are provided'''
    dir_ = os.getcwd()
    file_ = os.path.join(dir_, "master_pipeline.py")
    result = subprocess.run([file_, "../data/stackexchange/stack_exchange_data/corpus/anime", "-c", "-s"])
    assert result.returncode == 0