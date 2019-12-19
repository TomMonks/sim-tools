import numpy as np
import pytest

from ovs.fixed_budget import OCBA
from ovs.toy_models import guassian_bandit_sequence, BanditCasino

def test_ocba_update_mean_single():
    designs = guassian_bandit_sequence(1, 11)
    environment = BanditCasino(designs)

    optimiser = OCBA(model=environment,
                     n_designs=len(designs),
                     budget=1000,
                     delta=10)

    obs = np.arange(10)
    actual_mean = obs.mean()
    
    for x in obs:
        optimiser._allocations[0] += 1
        optimiser._update_moments(design_index=0,
                                  observation=x)
    
    assert actual_mean == optimiser._means[0]


def test_ocba_update_mean_multiple():
    designs = guassian_bandit_sequence(1, 11)
    environment = BanditCasino(designs)

    optimiser = OCBA(model=environment,
                     n_designs=len(designs),
                     budget=1000,
                     delta=10)
    n = 10
    obs = np.arange(n)
    actual_mean = np.full(n, -1, np.float)
    results = np.full(n, -1, np.float)
    
    for x in range(len(obs)):
        optimiser._allocations[0] += 1
        optimiser._update_moments(design_index=0,
                                  observation=obs[x])

        actual_mean[x] = obs[:x+1].mean()
        results[x] = optimiser._means[0]
        
    print(actual_mean)
    print(results)

    print(actual_mean.sum())
    print(results.sum())

    assert np.allclose(actual_mean, results)


def test_ocba_update_var():
    designs = guassian_bandit_sequence(1, 11)
    environment = BanditCasino(designs)

    optimiser = OCBA(model=environment,
                     n_designs=len(designs),
                     budget=1000,
                     delta=10)

    obs = np.arange(10)
    actual_mean = obs.mean()
    #sample var so divide by n - 1 for unbiased estimator
    actual_var = obs.var(ddof=1)

    for x in obs:
        optimiser._allocations[0] += 1
        optimiser._update_moments(design_index=0,
                                  observation=x)
    
    assert actual_var == optimiser._vars[0]


def test_ocba_update_var_one_obs():
    designs = guassian_bandit_sequence(1, 11)
    environment = BanditCasino(designs)

    optimiser = OCBA(model=environment,
                     n_designs=len(designs),
                     budget=1000,
                     delta=10)

    obs = np.arange(1)
    actual_mean = obs.mean()
    #only one vale so sample var = 0
    actual_var = 0

    for x in obs:
        optimiser._allocations[0] += 1
        optimiser._update_moments(design_index=0,
                                  observation=x)
    
    assert actual_var == optimiser._vars[0]


def test_ocba_update_var_multiple():
    designs = guassian_bandit_sequence(1, 11)
    environment = BanditCasino(designs)

    optimiser = OCBA(model=environment,
                     n_designs=len(designs),
                     budget=1000,
                     delta=10)

    n = 1000
    obs = np.arange(n)
    actual_var = np.full(n, -1, np.float)
    results = np.full(n, -1, np.float)
    
    for x in range(len(obs)):
        optimiser._allocations[0] += 1
        optimiser._update_moments(design_index=0,
                                  observation=obs[x])

        if x == 0:
            actual_var[0] = 0
        else:
            actual_var[x] = obs[:x+1].var(ddof=1)
        
        results[x] = optimiser._vars[0]
            
    assert (actual_var == results).all()


def test_ocba_update_var_multiple_random():
    designs = guassian_bandit_sequence(1, 11)
    environment = BanditCasino(designs)

    optimiser = OCBA(model=environment,
                     n_designs=len(designs),
                     budget=1000,
                     delta=10)

    n = 1000
    obs = np.random.normal(loc=10, scale=1, size=n)
    actual_var = np.full(n, -1, np.float)
    results = np.full(n, -1, np.float)
    
    for x in range(len(obs)):
        optimiser._allocations[0] += 1
        optimiser._update_moments(design_index=0,
                                  observation=obs[x])

        if x == 0:
            actual_var[0] = 0
        else:
            actual_var[x] = obs[:x+1].var(ddof=1)
        
        results[x] = optimiser._vars[0]
    
    print(actual_var.sum())
    print(results.sum())
    assert np.allclose(actual_var, results)


def test_ocba_allocate():
    designs = guassian_bandit_sequence(1, 5)
    environment = BanditCasino(designs)

    optimiser = OCBA(model=environment,
                     n_designs=5,
                     budget=400,
                     delta=100,
                     min=True,
                     n_0=4)

    optimiser._means = np.array([1.2, 2.1, 3.4, 4.87, 6.05])
    optimiser._vars = np.array([3.3, 2.0, 4.5, 5.3, 6.9])
    optimiser._allocations = np.array([12, 6, 5, 5, 4])
    
    actual_allocations = optimiser._allocate()
    expected_allocations = np.array([48, 38, 11, 2, 1])

    print(actual_allocations)
    print(expected_allocations)

    assert np.allclose(actual_allocations, expected_allocations)


    
    
