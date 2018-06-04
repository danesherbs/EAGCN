import numpy as np
import eagcn_pytorch.utils as shang_datasets
from datasets import lipophilicity, tox21, freesolv, hiv


def test_lipophilicity():
    theirs = shang_datasets.load_data("lipo", path='../data/')
    ours = lipophilicity.load_data()
    assert np.array_equal(theirs, ours)


def test_tox21():
    theirs = shang_datasets.load_data("tox21", path='../data/')
    ours = tox21.load_data()
    assert np.array_equal(theirs, ours)


def test_hiv():
    theirs = shang_datasets.load_data("hiv", path='../data/')
    ours = hiv.load_data()
    assert np.array_equal(theirs, ours)


def test_freesolv():
    theirs = shang_datasets.load_data("freesolv", path='../data/')
    ours = freesolv.load_data()
    assert np.array_equal(theirs, ours)
