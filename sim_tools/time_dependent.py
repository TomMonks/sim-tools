"""
Classes and functions to support time dependent samplingm in DES models.
"""

import itertools
import numpy as np

from typing import Optional


class NSPPThinning:
    """
    Non Stationary Poisson Process via Thinning.

    Thinning is an acceptance-rejection approach to sampling
    inter-arrival times (IAT) from a time dependent distribution
    where each time period follows its own exponential distribution.

    There are two random variables employed in sampling: an exponential
    distribution (used to sample IAT) and a uniform distibution (used
    to accept/reject samples).

    All IATs are sampled from an Exponential distribution with the highest
    arrival rate (most frequent). These arrivals are then rejected (thinned)
    proportional to the ratio of the current arrival rate to the maximum
    arrival rate.  The algorithm executes until a sample is accepted. The IAT
    returned is the sum of all the IATs that were sampled.

    """

    def __init__(
        self,
        data,
        random_seed1: Optional[int] = None,
        random_seed2: Optional[int] = None,
    ):
        """
        Non Stationary Poisson Process via Thinning.

        Time dependency is handled for a single table
        consisting of equally spaced intervals.

        Params:
        ------
        data: pandas.DataFrame
            list of time points during a period for transition between rates
            and list arrival rates in that period. Labels should be "t"
            and "arrival_rate" respectively.

        random_seed1: int, optional (default=None)
            Random seed for exponential distribution

        random_seed2: int
            Random seed for the uniform distribution used
            for acceptance/rejection sampling.
        """
        self.data = data
        self.arr_rng = np.random.default_rng(random_seed1)
        self.thinning_rng = np.random.default_rng(random_seed2)
        self.lambda_max = data["arrival_rate"].max()
        # assumes all other intervals are equal in length.
        self.interval = int(data.iloc[1]["t"] - data.iloc[0]["t"])
        self.rejects_last_sample = None

    def sample(self, simulation_time: float) -> float:
        """
        Run a single iteration of acceptance-rejection
        thinning alg to sample the next inter-arrival time

        Params:
        ------
        simulation_time: float
            The current simulation time.  This is used to look up
            the arrival rate for the time period.

        Returns:
        -------
        float
            The inter-arrival time
        """
        for _ in itertools.count():
            # this gives us the index of dataframe to use
            t = int(simulation_time // self.interval) % len(self.data)
            lambda_t = self.data["arrival_rate"].iloc[t]

            # set to a large number so that at least 1 sample taken!
            u = np.Inf

            # included for audit and tracking purposes.
            self.rejects_last_sample = 0.0

            interarrival_time = 0.0

            # reject samples if u >= lambda_t / lambda_max
            while u >= (lambda_t / self.lambda_max):
                self.rejects_last_sample += 1
                interarrival_time += self.arr_rng.exponential(1 / self.lambda_max)
                u = self.thinning_rng.uniform(0.0, 1.0)

            return interarrival_time
