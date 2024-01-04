"""
Convenient encapsulation of distributions 
and sampling from distributions not directly 
available in scipy or numpy. 

Useful for simulation.

Each distribution has its own random number stream
that can be set by a seed.

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
        super().__init__(random_seed)
        self.lower_bounds = np.asarray(lower_bounds)
        self.upper_bounds = np.asarray(upper_bounds)
        self.cumulative_probs = self.create_cumulative_probs(freq)

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
        freq = np.asarray(freq, dtype='float')
        return np.cumsum(freq / freq.sum(), dtype='float')

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

            # Obtain lower and upper bounds of a sample from the
            # discrete empirical distribution
            idx = np.searchsorted(self.cumulative_probs, U)
            lb, ub = self.lower_bounds[idx], self.upper_bounds[idx]

            # Use linear interpolation of U between
            # the lower and upper bound to obtain a continuous value
            continuous_value = lb + (ub - lb) * (U - self.cumulative_probs[idx - 1]) / (
                self.cumulative_probs[idx] - self.cumulative_probs[idx - 1]
            )

            samples.append(continuous_value)

        if size == 1:
            # .item() ensure returned as python 'float' 
            # as opposed to np.float64
            return samples[0].item()
        else:
            return np.asarray(samples)


class Erlang(Distribution):
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
        super().__init__(random_seed)
        self.mean = mean
        self.stdev = stdev
        self.location = location

        # k also referred to as shape
        self.k = round((mean / stdev) ** 2)

        # theta also referred to as scale
        self.theta = mean / self.k

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
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

        super().__init__(random_seed)
        self.shape = alpha
        self.scale = beta

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
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

        super().__init__(random_seed)
        self.alpha = alpha  # shape
        self.beta = beta  # scale
        self.location = location

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

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
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
        super().__init__(random_seed)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.min = lower_bound
        self.max = upper_bound

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
        """
        Sample fron the Beta distribution

        Params:
        -------
        size: int, optional (default=None)
            Number of samples to return. If integer then
            numpy array returned.
        """
        return self.min + (
            (self.max - self.min) * self.rng.beta(self.alpha1, self.alpha2, size)
        )


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

        super().__init__(random_seed)
        self.values = np.asarray(values)
        self.freq = np.asarray(freq)
        self.probabilities = self.freq / self.freq.sum()

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
        """
        Sample fron the Discrete distribution

        Params:
        -------
        size: int, optional (default=None)
            Number of samples to return. If integer then
            numpy array returned.
        """
        return self.rng.choice(self.values, p=self.probabilities, size=size).item()


class TruncatedDistribution(Distribution):
    """
    Truncated Distribution

    Pass in any distribution class and this class
    will tuncate the distribution at a lower bound.

    No resampling is done the class simply returns
    the maximum value.
    """

    def __init__(self, dist_to_truncate: Distribution, lower_bound: float):
        """
        Truncated distribution

        Params:
        -------

        dist_to_truncate: Distribution
            Any Distribution object that generates samples

        lower_bound: float
            Truncation point
        """
        self.dist = dist_to_truncate
        self.lower_bound = lower_bound

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
        """
        Sample fron the Discrete distribution

        Params:
        -------
        size: int, optional (default=None)
            Number of samples to return. If integer then
            numpy array returned.
        """
        if size is not None:
            samples = self.dist.sample(size)
            samples[samples < self.lower_bound] = self.lower_bound
            return samples

        else:
            sample = self.dist.sample()
            return max(self.lower_bound, sample)


class RawEmpirical(Distribution):
    """
    Sample with replacement from a list of raw empirical values

    Useful if none of the theoretical distributions on offer fit the data

    Notes:
    -----
    If sample size is small consider if upper and lower limits in raw data
    are representative of the real world system.
    """

    def __init__(self, values: npt.ArrayLike, random_seed: Optional[int] = None):
        """
        RawEmpirical

        Params:
        ------
        values: array-like
            Empirical list of sample values

        random_seed: int, optional (default=None)
            A random seed to reproduce samples. If set to none then a unique
            sample is created.
        """
        super().__init__(random_seed)
        self.values = np.asarray(values)

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
        """
        Sample from the raw empirical data with replacement

        Params:
        -------
        size: int, optional (default=None)
            Number of samples to return. If integer then
            numpy array returned.
        """
        return self.rng.choice(self.values, size)


class PearsonV(Distribution):
    """
    The PearsonV(alpha, beta) is an inverse Gamma distribution.

    Where alpha = shape, and beta = scale (> 0)

    Law (2007, pg 293-294) defines the distribution as
    PearsonV(alpha, beta) = 1/Gamma(alpha, 1/beta) and note that the
    PDF is similar to that of lognormal, but has a larger spike
    close to 0.  It can be used to model the time to complete a task.

    For certain values of the shape parameter the mean and var can be
    directly computed

    mean = beta / (alpha - 1) for alpha > 1.0
    var = beta^2 / (alpha - 1)^2 X (alpha - 2) fpr alpha > 2.0

    Alternative Sources:
    --------------------
    [1] https://riskwiki.vosesoftware.com/PearsonType5distribution.php
    [2] https://modelassist.epixanalytics.com/display/EA/Pearson+Type+5

    sources last accessed 03/01/2024

    Notes:
    ------
    A good R package for Pearson distributions is PearsonDS
    https://www.rdocumentation.org/packages/PearsonDS/versions/1.3.0

    """

    def __init__(self, alpha: float, beta: float, random_seed: Optional[int] = None):
        """
        PearsonV

        Params:
        ------
        alpha: float
            Shape parameter. Must be > 0

        beta: float
            Scale parameter. Must be > 0

        random_seed, int, optional (default=None)
            A random seed to reproduce samples. If set to none then a unique
            sample is created.
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be > 0")

        super().__init__(random_seed)
        self.alpha = alpha  # shape
        self.beta = beta  # scale

    def mean(self) -> float:
        """
        Compute the mean of the PearsonV

        If alpha <= 1.0 raises a ValueError
        """
        if self.alpha > 1.0:
            return self.beta / (self.alpha - 1)
        else:
            msg = "Cannot directly compute mean when alpha <= 1.0"
            raise ValueError(msg)

    def var(self) -> float:
        """
        Compute the Variance of the PearsonV

        If alpha <= 2.0 raises a ValueError
        """
        if self.alpha > 2.0:
            return (self.beta**2) / (((self.alpha - 1) ** 2) * (self.alpha - 2))
        else:
            msg = "Cannot directly compute var when alpha <= 2.0"
            raise ValueError(msg)

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
        """
        Sample from the PearsonV distribution

        Params:
        -------
        size: int, optional (default=None)
            Number of samples to return. If integer then
            numpy array returned.
        """
        return 1 / self.rng.gamma(self.alpha, 1 / self.beta, size)


class PearsonVI(Distribution):
    """
    The PearsonVI(alpha1, alpha2, beta) is an inverted beta distribution.

    Where:

    alpha1 = shape param 1, (> 0)
    alpha2 = shape param 2 (> 0)
    beta = scale (> 0)

    Law (2007, pg 294-295) notes that PearsonVI
    can be used to model the time to complete a task.

    For certain values of the 2nd shape parameter the mean and var can be
    directly computed. See functions mean() and var()

    Sampling:
    --------
    Pearson6(a1,a2,b) = b*X / (1-X), where X=Beta(a1,a2)

    Sources:
    --------------------
    [1] https://riskwiki.vosesoftware.com/PearsonType6distribution.php

    sources last accessed 03/01/2024

    Notes:
    ------
    A good R package for Pearson distributions is PearsonDS
    https://www.rdocumentation.org/packages/PearsonDS/versions/1.3.0

    """

    def __init__(
        self,
        alpha1: float,
        alpha2: float,
        beta: float,
        random_seed: Optional[int] = None,
    ):
        """
        PerasonVI

        Params:
        -------
        alpha1: float
            Shape parameters 1. Must be > 0

        alpha2: float
            Shape parameter 2. Must be > 0

        beta: float
            scale parameters. Must be > 0

        random_seed, int, optional (default=None)
            A random seed to reproduce samples. If set to none then a unique
            sample is created.
        """
        super().__init__(random_seed)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta

    def mean(self) -> float:
        if self.alpha2 > 1.0:
            return (self.beta * self.alpha1) / (self.alpha2 - 1)
        else:
            raise ValueError("Cannot compute mean when alpha2 <= 1.0")

    def var(self) -> float:
        """
        Compute the Variance of the PearsonV

        If alpha2 <= 2.0 raises a ValueError
        """
        if self.alpha2 > 2.0:
            return (
                (self.beta**2) * self.alpha1 * (self.alpha1 + self.alpha2 - 1)
            ) / (((self.alpha2 - 1) ** 2) * (self.alpha2 - 2))
        else:
            msg = "Cannot directly compute var when alpha2 <= 2.0"
            raise ValueError(msg)

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
        """
        Sample from the PearsonVI distribution

        Params:
        -------
        size: int, optional (default=None)
            Number of samples to return. If integer then
            numpy array returned.
        """
        # Pearson6(a1,a2,b)=b∗X/(1−X), where X=Beta(a1,a2,1)
        X = self.rng.beta(self.alpha1, self.alpha2, size)
        return self.beta * X / (1 - X)


class ErlangK(Distribution):
    """
    Erlang distribution where k and theta are specified.

    The Erlang is a special case of the gamma distribution where
    k is a positive integer.  Internally this is implemented using
    numpy Generators gamma method.

    Optionally a user can offet the original of the distribution
    using the location parameter.
    """

    def __init__(
        self,
        k: int,
        theta: float,
        location: Optional[float] = 0.0,
        random_seed: Optional[int] = None,
    ):
        """
        Constructor method

        Params:
        -------
        k: integer
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
        if k < 0.0:
            raise ValueError("k must be > 0")

        super().__init__(random_seed)
        self.k = k
        self.theta = theta
        self.location = location

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
        """
        Sample fron the Erlang distribution

        Params:
        -------
        size: int, optional (default=None)
            Number of samples to return. If integer then
            numpy array returned.
        """
        return self.rng.gamma(self.k, self.theta, size) + self.location


class Poisson(Distribution):
    """
    Poisson distribution

    Used to simulate number of events that occur in an interval of time.
    E.g. number of items in a batch.

    Sources:
    --------
    [1]  Law (2007 pg. 308) Simulation modelling and analysis.
    """

    def __init__(self, rate: float, random_seed: Optional[int] = None):
        """
        Poisson Distribution

        Params:
        -------
        rate: float
            Mean number of events in time period

        random_seed, int, optional (default=None)
            A random seed to reproduce samples. If set to none then a unique
            sample is created.
        """

        super().__init__(random_seed)
        self.rate = rate

    def sample(self, size: Optional[int] = None) -> float | np.ndarray:
        """
        Sample fron the Erlang distribution

        Params:
        -------
        size: int, optional (default=None)
            Number of samples to return. If integer then
            numpy array returned.
        """
        return self.rng.poisson(self.rate, size)
