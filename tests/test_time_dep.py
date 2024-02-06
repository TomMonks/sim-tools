"""
basic tests for time dependent module
"""

import sim_tools.time_dependent as td
from sim_tools.datasets import load_banks_et_al_nspp

import pytest

SEED_1 = 42
SEED_2 = 101

def test_sample_from_nspp():
    data = load_banks_et_al_nspp()
    d = td.NSPPThinning(data, SEED_1, SEED_2)
    assert type(d.sample()) == float