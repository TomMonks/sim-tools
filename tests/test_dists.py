
'''
basic smoke tests
Create objects to check all okay
'''
import sim_tools.distributions as dists

SEED_1 = 42

def test_exponential():
    d = dists.Exponential(10, random_seed=SEED_1)
    assert type(d.sample()) == float

def test_lognormal():
    d = dists.Lognormal(10, 1, random_seed=SEED_1)
    assert type(d.sample()) == float

def test_normal():
    d = dists.Normal(10, 1, random_seed=SEED_1)
    assert type(d.sample()) == float

def test_uniform():
    d = dists.Uniform(1, 10, random_seed=SEED_1)
    assert type(d.sample()) == float

def test_tri():
    d = dists.Triangular(1.0, 10.0, 25.0, random_seed=SEED_1)
    assert type(d.sample()) == float

def test_bernoulli():
    d = dists.Bernoulli(0.3, random_seed=SEED_1)
    assert type(d.sample()) == int

def test_fixed_type():
    d = dists.FixedDistribution(5.0)
    assert type(d.sample()) == float

def test_fixed_value():
    d = dists.FixedDistribution(5.0)
    assert d.sample() == 5.0

def test_combination():
    foo = dists.Exponential(10, random_seed=SEED_1)
    bar = dists.Normal(10, 1, random_seed=SEED_1)
    d = dists.CombinationDistribution(foo, bar)
    sample = d.sample()
    assert type(sample) == float

