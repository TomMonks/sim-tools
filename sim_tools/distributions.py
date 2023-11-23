"""
Convenient packaging of distributions for simulation.

Each distribution has its own random number stream.

"""

from abc import ABC, abstractmethod
import math
import numpy as np


class Distribution(ABC):
    """
    Distribution abstract class
    All distributions derived from it.
    """

    def __init__(self, random_seed=None):
        self.rng = np.random.default_rng(random_seed)

    @abstractmethod
    def sample(self, size=None):
        """
        Generate a sample from the distribution

        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.

        Returns:
        -------
        np.array or scalar
        """
        pass


class Exponential(Distribution):
    """
    Convenience class for the exponential distribution.
    packages up distribution parameters, seed and random generator.
    """

    def __init__(self, mean, random_seed=None):
        """
        Constructor

        Params:
        ------
        mean: float
            The mean of the exponential distribution

        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        """
        super().__init__(random_seed)
        self.mean = mean

    def sample(self, size=None):
        """
        Generate a sample from the exponential distribution

        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        """
        return self.rng.exponential(self.mean, size=size)


class Bernoulli(Distribution):
    """
    Convenience class for the Bernoulli distribution.
    packages up distribution parameters, seed and random generator.
    """

    def __init__(self, p, random_seed=None):
        """
        Constructor

        Params:
        ------
        p: float
            probability of drawing a 1

        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        """
        super().__init__(random_seed)
        self.p = p

    def sample(self, size=None):
        """
        Generate a sample from the exponential distribution

        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        """
        return self.rng.binomial(n=1, p=self.p, size=size)


class Lognormal(Distribution):
    """
    Encapsulates a lognormal distirbution
    """

    def __init__(self, mean, stdev, random_seed=None):
        """
        Params:
        -------
        mean: float
            mean of the lognormal distribution

        stdev: float
            standard dev of the lognormal distribution

        random_seed: int, optional (default=None)
            Random seed to control sampling
        """
        super().__init__(random_seed)
        mu, sigma = self.normal_moments_from_lognormal(mean, stdev**2)
        self.mu = mu
        self.sigma = sigma

    def normal_moments_from_lognormal(self, m, v):
        """
        Returns mu and sigma of normal distribution
        underlying a lognormal with mean m and variance v
        source: https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal
        -data-with-specified-mean-and-variance.html

        Params:
        -------
        m: float
            mean of lognormal distribution
        v: float
            variance of lognormal distribution

        Returns:
        -------
        (float, float)
        """
        phi = math.sqrt(v + m**2)
        mu = math.log(m**2 / phi)
        sigma = math.sqrt(math.log(phi**2 / m**2))
        return mu, sigma

    def sample(self):
        """
        Sample from the normal distribution
        """
        return self.rng.lognormal(self.mu, self.sigma)


class Normal(Distribution):
    """
    Convenience class for the normal distribution.
    packages up distribution parameters, seed and random generator.

    Option to prevent negative samples by resampling

    """

    def __init__(self, mean, sigma, allow_neg=True, random_seed=None):
        """
        Constructor

        Params:
        ------
        mean: float
            The mean of the normal distribution

        sigma: float
            The stdev of the normal distribution

        allow_neg: bool, optional (default=True)
            False = resample on negative values
            True = negative samples allowed.

        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        """
        super().__init__(random_seed)
        self.mean = mean
        self.sigma = sigma
        self.allow_neg = allow_neg

    def sample(self, size=None):
        """
        Generate a sample from the normal distribution

        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        """
        # initial sample
        samples = self.rng.normal(self.mean, self.sigma, size=size)

        # no need to check if neg allowed.
        if self.allow_neg:
            return samples

        # repeatedly resample negative values
        negs = np.where(samples < 0)[0]
        while len(negs) > 0:
            resample = self.rng.normal(self.mean, self.sigma, size=len(negs))
            samples[negs] = resample
            negs = np.where(samples < 0)[0]

        return samples


class Uniform(Distribution):
    """
    Convenience class for the Uniform distribution.
    packages up distribution parameters, seed and random generator.
    """

    def __init__(self, low, high, random_seed=None):
        """
        Constructor

        Params:
        ------
        low: float
            lower range of the uniform

        high: float
            upper range of the uniform

        random_seed: int, optional (default=None)
            A random seed to reproduce samples.  If set to none then a unique
            sample is created.
        """
        super().__init__(random_seed)
        self.low = low
        self.high = high

    def sample(self, size=None):
        """
        Generate a sample from the uniform distribution

        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        """
        return self.rng.uniform(low=self.low, high=self.high, size=size)


class Triangular(Distribution):
    """
    Convenience class for the triangular distribution.
    packages up distribution parameters, seed and random generator.
    """

    def __init__(self, low, mode, high, random_seed=None):
        super().__init__(random_seed)
        self.low = low
        self.high = high
        self.mode = mode

    def sample(self, size=None):
        return self.rng.triangular(self.low, self.mode, self.high, size=size)


class FixedDistribution(Distribution):
    """
    Simple fixed distribution.  Return scalar or numpy array
    of a fixed value.
    """

    def __init__(self, value):
        self.value = value

    def sample(self, size=None):
        """
        Generate a sample from the fixed distribution

        Params:
        ------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.
        """
        if size is not None:
            return np.full(size, self.value)
        else:
            return self.value


class CombinationDistribution(Distribution):
    """
    Simple summation of samples from multiple distributions.
    """

    def __init__(self, *dists):
        self.dists = dists

    def sample(self, size=None):
        total = 0.0 if size is None else np.zeros(size)

        for dist in self.dists:
            total += dist.sample(size)
        return total


if __name__ == "__main__":
    n = Normal(0, 1, allow_neg=False)
    samples = n.sample(10)
    print(samples)
