"""
basic smoke tests
Create objects to check all okay
"""
import sim_tools.distributions as dists
import pytest

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


def test_erlang():
    d = dists.Erlang(10.0, 2.8, random_seed=SEED_1)
    assert type(d.sample()) == float


def test_erlangk():
    d = dists.ErlangK(1, 2.8, random_seed=SEED_1)
    assert type(d.sample()) == float


def test_gamma():
    d = dists.Gamma(1.2, 2.8, random_seed=SEED_1)
    assert type(d.sample()) == float


def test_weibull():
    d = dists.Weibull(1.2, 2.8, random_seed=SEED_1)
    assert type(d.sample()) == float


def test_beta():
    d = dists.Beta(1.2, 2.8, random_seed=SEED_1)
    assert type(d.sample()) == float


def test_pearsonv():
    d = dists.PearsonV(1.2, 2.8, random_seed=SEED_1)
    assert type(d.sample()) == float


def test_pearsonvi():
    d = dists.PearsonVI(1.2, 1.2, 2.8, random_seed=SEED_1)
    assert type(d.sample()) == float


def test_poisson():
    d = dists.Poisson(5.5, random_seed=SEED_1)
    assert type(d.sample()) == int


def test_continuous_empirical():
    dist = dists.ContinuousEmpirical(
        lower_bounds=[0, 5, 10, 15, 30, 45, 60, 120, 180, 240, 480],
        upper_bounds=[5, 10, 15, 30, 45, 60, 120, 180, 240, 480, 2880],
        freq=[34, 4, 8, 13, 15, 13, 19, 13, 9, 12, 73],
        random_seed=SEED_1,
    )
    assert type(dist.sample()) == float


def test_continous_empirical_length():
    dist = dists.ContinuousEmpirical(
        lower_bounds=[0, 5, 10, 15, 30, 45, 60, 120, 180, 240, 480],
        upper_bounds=[5, 10, 15, 30, 45, 60, 120, 180, 240, 480, 2880],
        freq=[34, 4, 8, 13, 15, 13, 19, 13, 9, 12, 73],
        random_seed=SEED_1,
    )
    expected_size = 10

    assert len(dist.sample(expected_size)) == expected_size


def test_discrete():
    d = dists.Discrete(values=[1, 2, 3], freq=[95, 3, 2], random_seed=SEED_1)
    assert type(d.sample()) == int


def test_discrete():
    d = dists.Discrete(values=[1, 2, 3], freq=[95, 3, 2], random_seed=SEED_1)
    assert type(d.sample()) == int


def test_truncated_type():
    d1 = dists.Normal(10, 1, random_seed=SEED_1)
    d2 = dists.TruncatedDistribution(d1, lower_bound=10.0)
    assert type(d2.sample()) == float


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, 10.0),
        (10, 10.0),
        (100, 10.0),
        (10_000_000, 10.0),
        (10_000_000, 0.0),
        (10_000_000, 0.01),
    ],
)
def test_truncated_min(n, expected):
    d1 = dists.Normal(10, 1, random_seed=SEED_1)
    d2 = dists.TruncatedDistribution(d1, lower_bound=expected)
    assert min(d2.sample(size=n)) >= expected
