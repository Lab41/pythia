'''
tests master pipeline module

system tests checking correct arguments. tested from pythia directory.
'''

import pytest
from src.pipelines import master_pipeline
# import subprocess
# import os

# def test_no_directory():
#     '''tests that pipeline quits with the correct exit code if no directory is included'''
#     result = subprocess.run(['src/pipelines/master_pipeline.py'])
#     assert result.returncode == 2

# def test_no_features():
#     '''tests that pipeline quits with the correct exit code if no features are included'''
#     result = subprocess.run(['src/pipelines/master_pipeline.py', 'dir'])
#     assert result.returncode == 1

# def test_no_algorithms():
#     '''tests that pipeline quits with the correct exit code if no algorithms are included'''
#     result = subprocess.run(['src/pipelines/master_pipeline.py', 'dir', '-c'])
#     assert result.returncode == 3

# def test_successful_run():
#     '''tests that pipeline has an exit code of zero (runs all the way through) when the right args are provided'''
#     result = subprocess.run(['src/pipelines/master_pipeline.py', "data/stackexchange/anime", "-c", "-s"])
#     assert result.returncode == 0

# Test that argument list for master_pipeline is same as in experiments.py
def test_variable_list():
    import experiments.experiments
    pipeline_argcount = master_pipeline.get_args.__code__.co_argcount
    pipeline_args = master_pipeline.get_args.__code__.co_varnames[:pipeline_argcount]
    experiment_argcount = experiments.experiments.run_experiment.__code__.co_argcount
    experiment_args = experiments.experiments.run_experiment.__code__.co_varnames[:experiment_argcount]
    # remove Sacred-specific arg if present
    try:
        experiment_args = tuple(experiment_args[:experiment_args.index('_run')] + experiment_args[experiment_args.index('_run') + 1:])
    except:
        pass
    assert pipeline_args == experiment_args
