import pytest
import numpy as np
import eagcn_pytorch.utils as shang_datasets
from datasets import lipophilicity, tox21, freesolv, hiv


def test_lipophilicity():
    shang_datasets.load_data("lipo", path='../data/')
    lipophilicity.load_data()


def test_tox21():
    shang_datasets.load_data("tox21", path='../data/')
    tox21.load_data()


def test_hiv():
    shang_datasets.load_data("hiv", path='../data/')
    hiv.load_data()


def test_freesolv():
    shang_datasets.load_data("freesolv", path='../data/')
    freesolv.load_data()
