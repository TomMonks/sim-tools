"""
Classes and functions to support time dependent samplingm in DES models.
"""

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, Tuple


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
        self.min_iat = data["mean_iat"].min()
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
            u = np.inf

            # included for audit and tracking purposes.
            self.rejects_last_sample = 0.0

            interarrival_time = 0.0

            # reject samples if u >= lambda_t / lambda_max
            while u >= (lambda_t / self.lambda_max):
                self.rejects_last_sample += 1
                interarrival_time += self.arr_rng.exponential(self.min_iat)
                u = self.thinning_rng.uniform(0.0, 1.0)

            return interarrival_time


def nspp_simulation(
    arrival_profile: pd.DataFrame,
    run_length: Optional[float] = None,
    n_reps: Optional[int] = 1000,
) -> pd.DataFrame:
    """
    Generate a pandas dataframe that contains multiple replications of
    a non-stationary poisson process for the set arrival profile.

    This uses the sim-tools NSPPThinning class.

    Useful for validating the the NSPP has been set up correctly and is producing the
    desired profile for the simulation model.

    On each replication the function counts the number of arrivals during the intervals
    from the arrival profile.  Returns a data frame with reps (rows) and interval arrivals
    (columns)

    Parameters:
    -----------
    arrival_profile: pandas.DataFrame
        The arrival profile is a pandas data frame containing 't', 'arrival_rate' and
        'mean_iat' columns.

    run_length: float, optional (default=None)
        How long should the simulation be run. If none then uses the last value in 't'
        + the interval (assumes equal width intervals)

    n_reps: int, optional (default=1000)
        The number of replications to run.

    Returns:
    --------
    pd.DataFrame.


    """
    # replication results
    replication_results = []

    # multiple replications
    for rep in range(n_reps):

        # method for producing n non-overlapping streams
        seed_sequence = np.random.SeedSequence(rep)

        # Generate n high quality child seeds
        seeds = seed_sequence.spawn(2)

        # create nspp
        nspp_rng = NSPPThinning(arrival_profile, seeds[0], seeds[1])

        # if no run length has been set....
        if run_length is None:
            run_length = (
                arrival_profile["t"].iloc[len(arrival_profile) - 1] + nspp_rng.interval
            )

        # list - each item is an interval in the arrival profile
        interval_samples = [0] * arrival_profile.shape[0]
        simulation_time = 0.0
        while simulation_time < run_length:
            iat = nspp_rng.sample(simulation_time)
            simulation_time += iat


            if simulation_time < run_length:
                # data collection: add one to count for hour of the day
                # note list NSPPThinning this assume equal intervals
                interval_of_day = (
                    int(simulation_time // nspp_rng.interval) % len(arrival_profile)
                )
                interval_samples[interval_of_day] += 1

        replication_results.append(interval_samples)

    # produce summary chart of arrivals per interval
    # format in a dataframe
    df_replications = pd.DataFrame(replication_results)
    df_replications.index = np.arange(1, len(df_replications) + 1)
    df_replications.index.name = "rep"

    return df_replications


def nspp_plot(
    arrival_profile: pd.DataFrame,
    run_length: Optional[float] = None,
    n_reps: Optional[int] = 1000,
) -> Tuple[plt.Figure, plt.Axes]:
    """Generate a matplotlib chart to visualise a non-stationary poisson process
    for the set arrival profile.

    This uses the sim-tools NSPPThinning class.

    Useful for validating the the NSPP has been set up correctly and is producing the
    desired profile for the simulation model.

    Parameters:
    ----------
    arrival_profile: pandas.DataFrame
        The arrival profile is a pandas data frame containing 't', 'arrival_rate' and
        'mean_iat' columns.

    run_length: float, optional (default=None)
        How long should the simulation be run. If none then uses the last value in 't'
        + the interval (assumes equal width intervals)

    n_reps: int, optional (default=1000)
        The number of replications to run.
    """

    # verification of arrival_profile

    # is it a dataframe
    if not isinstance(arrival_profile, pd.DataFrame):
        raise ValueError(
            f"arrival_profile expected pd.DataFrame " f"got {type(arrival_profile)}"
        )

    # all columns are present
    required_columns = ["t", "arrival_rate", "mean_iat"]
    for col in required_columns:
        if col not in arrival_profile.columns:
            raise ValueError(
                f"arrival_profile must contain "
                f"the following columns: {required_columns}. "
            )

    # generate the sample data
    df_interval_results = nspp_simulation(arrival_profile, run_length, n_reps)

    interval_means = df_interval_results.mean(axis=0)
    interval_sd = df_interval_results.std(axis=0)

    upper = interval_means + interval_sd
    lower = interval_means - interval_sd
    lower[lower < 0] = 0

    # visualise
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot()

    # chart x ticks
    x_values = np.arange(0, arrival_profile.shape[0])

    # plot in this case returns a 2D line plot object
    _ = ax.plot(arrival_profile["t"], interval_means, label="Mean")
    _ = ax.fill_between(arrival_profile["t"], lower, upper, alpha=0.2, label="+-1SD")

    # chart appearance
    _ = ax.legend(loc="best", ncol=3)
    _ = ax.set_ylim(
        0,
    )
    _ = ax.set_xlim(0, arrival_profile.shape[0] - 1)
    _ = ax.set_ylabel("arrivals")
    _ = ax.set_xlabel("interval (from profile)")
    _ = plt.xticks(arrival_profile["t"])

    return fig, ax
