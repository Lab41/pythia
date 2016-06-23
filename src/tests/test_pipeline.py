from src.pipelines import master_pipeline
import subprocess
import os

def test_no_directory():
    os.chdir('../pipelines')
    dir_ = os.getcwd()
    file_ = os.path.join(dir_, "master_pipeline.py")
    result = subprocess.run([file_])
    assert result.returncode == 2

def test_no_features():
    os.chdir('../pipelines')
    dir_ = os.getcwd()
    file_ = os.path.join(dir_, "master_pipeline.py")
    result = subprocess.run([file_, "dir"])
    assert result.returncode == 1

def test_no_algorithms():
    os.chdir('../pipelines')
    dir_ = os.getcwd()
    file_ = os.path.join(dir_, "master_pipeline.py")
    result = subprocess.run([file_, "dir", "-c"])
    assert result.returncode == 3

def test_successful_run():
    os.chdir('../pipelines')
    dir_ = os.getcwd()
    file_ = os.path.join(dir_, "master_pipeline.py")
    result = subprocess.run([file_, "dir", "-c", "-s"])
    assert result.returncode == 0