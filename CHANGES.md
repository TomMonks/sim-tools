# Change log

## v0.5.0

### Added

* EXPERIMENTAL: added `trace` module with `Traceable` class for colour coding output from different processes and tracking individual patients.

### Fixed

* DIST: fix to `NSPPThinning` sampling to pre-calcualte mean IAT to ensure that correct exponential mean is used.

## v0.4.0

* BUILD: Dropped legacy `setuptools` and migrated package build to `hatch`
* BUILD: Removed `setup.py`, `requirements.txt` and `MANIFEST` in favour of `pyproject.toml`

## v0.3.3

* PATCH: `distributions.Discrete` was not returning numpy arrays.

## v0.3.2

* Distributions classes now have python type hints.
* Added distributions and time dependent arrivals via thinning example notebooks.
* Added `datasets` module and function to load example NSPP dataset.
* Distributions added
    * Erlang (mean and stdev parameters)
    * ErlangK (k and theta parameters)
    * Poisson
    * Beta
    * Gamma
    * Weibull
    * PearsonV
    * PearsonVI
    * Discrete (values and observed frequency parameters)
    * ContinuousEmpirical (linear interpolation between groups)
    * RawEmpirical (resample with replacement from individual X's)
    * TruncatedDistribution (arbitrary truncation of any distribution)
* Added sim_tools.time_dependent module that contains `NSPPThinning` class for modelling time dependent arrival processes.
* Updated test suite for distributions and thinning
* Basic Jupyterbook of documentation.

## v0.2.0

* Added `sim_tools.distribution` module.  This contains classes representing popular sampling distributions for Discrete-event simulation. All classes encapsulate a `numpy.random.Generator` object, a random seed, and the parameters of a sampling distribution.  

* Python has been updated, tested, and patched for 3.10 and 3.11 as well as numpy 1.20+

* Minor linting and code formatting improvement.