"""
Convenient packaging of distributions for simulation.

Each distribution has its own random number stream.

"""

from abc import ABC, abstractmethod
import math
import numpy as np

from typing import Optional, Tuple
import numpy.typing as npt


class Distribution(ABC):
    """
    Distribution abstract class
    All distributions derived from it.
    """

    def __init__(self, random_seed: Optional[int] = None):
        self.rng = np.random.default_rng(random_seed)

    @abstractmethod
    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
        """
        Generate a sample from the distribution

        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.

        Returns:
        -------
        np.ndarray or scalar
        """
        pass


class Exponential(Distribution):
    """
    Convenience class for the exponential distribution.
    packages up distribution parameters, seed and random generator.
    """

    def __init__(self, mean: float, random_seed: Optional[int] = None):
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

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
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

    def __init__(self, p: float, random_seed: Optional[int] = None):
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

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
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

    def __init__(self, mean: float, stdev: float, random_seed: Optional[int] = None):
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

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
        """
        Sample from the normal distribution
        """
        return self.rng.lognormal(self.mu, self.sigma, size=size)


class Normal(Distribution):
    """
    Convenience class for the normal distribution.
    packages up distribution parameters, seed and random generator.

    Option to prevent negative samples by resampling

    """

    def __init__(
        self,
        mean: float,
        sigma: float,
        allow_neg: Optional[bool] = True,
        random_seed: Optional[int] = None,
    ):
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

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
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

    def __init__(
        self, low: float, high: float, random_seed: Optional[int] = None
    ) -> float | np.ndarray:
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

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
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

    def __init__(
        self, low: float, mode: float, high: float, random_seed: Optional[int] = None
    ) -> float | np.ndarray:
        super().__init__(random_seed)
        self.low = low
        self.high = high
        self.mode = mode

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
        return self.rng.triangular(self.low, self.mode, self.high, size=size)


class FixedDistribution(Distribution):
    """
    Simple fixed distribution.  Return scalar or numpy array
    of a fixed value.
    """

    def __init__(self, value: float):
        self.value = value

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
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

    def __init__(self, *dists: Distribution):
        self.dists = dists

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
        """
        Sample from the combination distribution

        Params:
        -------
        size: int, optional (default=None)
            the number of samples to return.  If size=None then a single
            sample is returned.

        Returns:
        -------
        np.ndarray or scalar
        """
        total = 0.0 if size is None else np.zeros(size)

        for dist in self.dists:
            total += dist.sample(size)
        return total


class ContinuousEmpirical(Distribution):
    """
    Continuous Empirical Distribution.

    Linear interpolation between upper and lower
    bounds of a discrete distribution
    """

    def __init__(
        self,
        lower_bounds: npt.ArrayLike,
        upper_bounds: npt.ArrayLike,
        freq: npt.ArrayLike,
        random_seed: Optional[int] = None,
    ):
        """
        Continuous Empirical Distribution.

        Params:
        ------
        lower_bounds: array-like
            Lower bounds of a discrete empirical distribution

        upper_bounds: array-like
            Upper bounds of a discrete empirical distribution

        freq: array-like
            Frequency of observations between bounds

        random_seed: int, optional (default=None)
            A random seed to reproduce samples. If set to none then a unique
            sample is created.

        """
        self.lower_bounds = np.asarray(lower_bounds)
        self.upper_bounds = np.asarray(upper_bounds)
        self.cumulative_probs = self.create_cumulative_probs(freq)
        self.rng = np.random.default_rng(random_seed)

    def create_cumulative_probs(self, freq: npt.ArrayLike) -> npt.NDArray[float]:
        """
        Calculate cumulative relative frequency from
        frequency

        Params:
        ------
        freq: array-like
            frequency distribution

        Returns:
        --------
        np.ndarray
            Cumulative relative frequency.
        """
        freq = np.asarray(freq)
        return np.cumsum(freq / freq.sum())

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
        """
        Sample fron the Continuous Empirical Distribution
        function.

        Params:
        -------
        size: int, optional (default=None)
            Number of samples to return. If integer then
            numpy array returned.
        """
        if size is None:
            size = 1

        samples = []
        for i in range(size):
            # Sample a value U from the uniform(0, 1) distribution
            U = self.rng.random()

            # Obtain lower and upper bounds of a sample from the discrete empirical distribution
            idx = np.searchsorted(self.cumulative_probs, U)
            lb, ub = self.lower_bounds[idx], self.upper_bounds[idx]

            # Use linear interpolation of U between the lower and upper bound to obtain a continuous value
            continuous_value = lb + (ub - lb) * (U - self.cumulative_probs[idx - 1]) / (
                self.cumulative_probs[idx] - self.cumulative_probs[idx - 1]
            )

            samples.append(continuous_value)

        if size == 1:
            return samples[0]
        else:
            return np.asarray(samples)


class Erlang:
    """
    Erlang distribution

    Implemented to allow for users to input the mean,
    and stdev of the distribution as opposed to k and theta.

    Mean and stdev are convered to k and theta internally.

    The Erlang is a special case of the gamma distribution where
    k is an integer.  Internally this is implemented using
    numpy Generators gamma method.  As k is calculated from the mean
    and stdev it is rounded to an integer value using python's
    built in 'round' function.

    Optionally a user can offet the original of the distribution
    using the location parameter.

    Sources:
    -------
    convert between mean+stdev to k+theta:
    https://www.statisticshowto.com/erlang-distribution/
    """

    def __init__(
        self,
        mean: float,
        stdev: float,
        location: Optional[float] = 0.0,
        random_seed: Optional[int] = None,
    ):
        """
        Consructor method

        Params:
        -------
        mean: float
            Mean of the Erlang

        stdev: float
            Standard deviation of the Erlang distribution

        location: float, optional (default=0.0)
            Offset the original of the distribution i.e.
            the returned value = sample[Erlang] + location

        random_seed, int, optional (default=None)
            A random seed to reproduce samples. If set to none then a unique
            sample is created.
        """
        self.mean = mean
        self.stdev = stdev
        self.location = location
        self.rng = np.random.default_rng(random_seed)

        # k also referred to as shape
        self.k = round((mean / stdev) ** 2)

        # theta also referred to as scale
        self.theta = mean / self.k

    def sample(self, size=None):
        """
        Sample fron the Erlang distribution

        Params:
        -------
        size: int, optional (default=None)
            Number of samples to return. If integer then
            numpy array returned.
        """
        return self.rng.gamma(self.k, self.theta, size) + self.location


class Weibull(Distribution):
    """
    Weibull distribution

    The Weibull takes shape (alpha) and scale (beta) parameters.  Both shape and scale
    should be > 0. The higher the scale parameters the more variance in the samples.

    This implementation also includes a third parameter "location"
    (default = 0) to shift the distribution if a lower bound is needed.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        location: Optional[int] = 0.0,
        random_seed: Optional[int] = None,
    ):
        """
        Three parameter Weibull distribution.

        Params:
        ------
        alpha: float
            The shape parameter.

        beta: float:
            The scale parameter.  The higher the scale parameters the
            more variance in the samples

        location: float, optional (default=None)
            An offset from 0.0

        Notes:
        ------
        Check that the mean and variance of the samples are as expected.

        This is because it is easy to make mistakes when setting the shape
        and scale parameters if converting from other notation.
        For example:

        In Law and Kelton, shape=alpha and scale=beta. But ->

        Wikipedia defines shape=k and scale = lambda = 1/beta
        https://en.wikipedia.org/wiki/Weibull_distribution

        other sources define shape=beta and scale=eta (η)

        In random.weibullvariate alpha=scale and beta=shape!
        """

        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be > 0")

        self.shape = alpha
        self.scale = beta
        self.rng = np.random.default_rng(random_seed)

    def sample(self, size: Optional[int] = None):
        """
        Sample fron the Erlang distribution

        Params:
        -------
        size: int, optional (default=None)
            Number of samples to return. If integer then
            numpy array returned.
        """
        return self.scale * self.rng.weibull(self.shape, size)


class Gamma(Distribution):
    """
    Gamma distribution

    Gamma distribution set up to accept alpha (scale) and beta (shape)
    parameters as described in Law (2007).

    Also contains functions to compute mean, variance, and a static method
    to computer alpha and beta from specified mean and variance.

    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        location: Optional[float] = 0.0,
        random_seed: Optional[int] = None,
    ):
        """
        Gamma distribution

        Params:
        ------
        alpha: float. Must be > 0

        beta: float
            scale parameter. Must be > 0

        location, float, optional (default=0.0)
            Offset the original of the distribution i.e.
            the returned value = sample[Gamma] + location

        random_seed: int, optional (default=None)
            A random seed to reproduce samples. If set to none then a unique
            sample is created.

        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be > 0")

        self.alpha = alpha  # shape
        self.beta = beta  # scale
        self.location = location
        self.rng = np.random.default_rng(random_seed)

    def mean(self) -> float:
        """
        The computed mean of the gamma distribution

        Returns:
        -------
        float
        """
        return self.alpha * self.beta

    def variance(self) -> float:
        """
        The computed varaince of the gamma distribution

        Returns:
        -------
        float
        """
        return self.alpha * (self.beta**2)

    @staticmethod
    def params_from_mean_and_var(mean: float, var: float) -> Tuple[float, float]:
        """
        Helper static method to get alpha and beta parameters
        from a mean and variance.

        Params:
        ------
        mean: float
            mean of the gamma distribution

        var: float
            variance of the gamma distribution

        Returns:
        -------
        (float, float)
        alpha, beta

        """
        alpha = mean**2 / var
        beta = mean / var
        return alpha, beta

    def sample(self, size: Optional[int] = None):
        """
        Sample fron the Gamma distribution

        Params:
        -------
        size: int, optional (default=None)
            Number of samples to return. If integer then
            numpy array returned.
        """
        return self.rng.gamma(self.alpha, self.beta, size) + self.location


class Beta(Distribution):
    """
    Beta distribution

    As defined in  Simulation Modeling and Analysis (Law, 2007).

    Accepts to shape parameters alpha1 and alpha2.
    The beta distribution is [0, 1].
    This can be rescaled using to [min, max] using
    min + (max - min) * sample(Beta)

    Common uses:
    -----------
    1. Useful as a rough model in the absence data
    2. Distribution of a random proportion
    3. Time to complete a task.
    """

    def __init__(
        self,
        alpha1: float,
        alpha2: float,
        lower_bound: Optional[float] = 0.0,
        upper_bound: Optional[float] = 1.0,
        random_seed: Optional[int] = None,
    ):
        """
        Beta distribution

        Params:
        -------
        alpha1: float
            shape parameter 1

        alpha2: float
            shape parameter 2

        min: float, optional (default=0.0)
            Used with max to rescale [0,1] to [min, max]

        max: float, optional (default=1.0)
            Used with max to rescale [0,1] to [min, max]

        random_seed: int, optional (default=None)
            A random seed to reproduce samples. If set to none then a unique
            sample is created.
        """
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.min = lower_bound
        self.max = upper_bound
        self.rng = np.random.default_rng(random_seed)

    def sample(self, size: Optional[int] = None):
        """
        Sample fron the Beta distribution

        Params:
        -------
        size: int, optional (default=None)
            Number of samples to return. If integer then
            numpy array returned.
        """
        return self.min + ((self.max - self.min) * self.rng.beta(
            self.alpha1, self.alpha2, size
        ))


class Discrete(Distribution):
    """
    Discrete distribution

    Sample a value with a given observed frequency.

    Example uses:
    -------------
    1. routing percentages
    2. classes of entity
    3. batch sizes of arrivals
    """

    def __init__(
        self,
        values: npt.ArrayLike,
        freq: npt.ArrayLike,
        random_seed: Optional[int] = None,
    ):
        """
        Discrete distribution

        Params:
        ------
        values: array-like
            list of sample values. Must be of equal length to freq

        freq: array-like
            list of observed frequencies. Must be of equal length to values

        random_seed, int, optional (default=None)
            A random seed to reproduce samples. If set to none then a unique
            sample is created.
        """
        if len(values) != len(freq):
            raise ValueError("values and freq arguments must be of equal length")

        self.values = np.asarray(values)
        self.freq = np.asarray(freq)
        self.probabilities = self.freq / self.freq.sum()
        self.rng = np.random.default_rng(random_seed)

    def sample(self, size: Optional[int] = None):
        """
        Sample fron the Discrete distribution

        Params:
        -------
        size: int, optional (default=None)
            Number of samples to return. If integer then
            numpy array returned.
        """
        return self.rng.choice(self.values, p=self.probabilities, size=size)
