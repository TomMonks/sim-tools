# Change log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Dates formatted as YYYY-MM-DD as per [ISO standard](https://www.iso.org/iso-8601-date-and-time-format.html).

Consistent identifier (represents all versions, resolves to latest): [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4553641.svg)](https://doi.org/10.5281/zenodo.4553641)


## [v0.7.1](https://github.com/TomMonks/sim-tools/releases/tag/v0.7.1)  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14844701.svg)](https://doi.org/10.5281/zenodo.14844701)

### Fixed

* Patched `ReplicationsAlgorithm` look ahead will now correctly use `_klimit()` to calculate extra no. replications to run.

## [v0.7.0](https://github.com/TomMonks/sim-tools/releases/tag/v0.7.0) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14834956.svg)](https://doi.org/10.5281/zenodo.14834956)

### Added

* `output_analysis` module - focussed at the moment on selecting the number of replications
* `ReplicationsAlgorithm` that implements the automated approach to selecting the number of replications for a single performance measures.
* `ReplicationsAlgorithmModelAdapter` - a `Protocol` to adapt any model to work with with `ReplicationsAlgorithm`
* `confidence_interval_method` - select the number of replication using the classical confidence interval method
* `plotly_confidence_interval_method` - visualise the confidence interval method using plotly.
* `ReplicationObserver` a `Protocol` for observering the replications algorithm 
*  `ReplicationTabulizer` record replications algorithm in a pandas dataframe. 
* Documentation for `ReplicationsAlgorithm`

### Updated

* `sim-tools` dev conda environment now pip installs local python package in editable model.

## [v0.6.1](https://github.com/TomMonks/sim-tools/releases/tag/v0.6.1) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13135700.svg)](https://doi.org/10.5281/zenodo.13135700)

### Fixed

* BUILD: added rich library. 

### Removed

* Scipy Dependency

## [v0.6.0](https://github.com/TomMonks/sim-tools/releases/tag/v0.6.0) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13122484.svg)](https://doi.org/10.5281/zenodo.13122484)

### Added

* Added `nspp_plot` and `nspp_simulation` functions to `time_dependent` module.
* DOCS: added `nspp_plot` and `nspp_simulation` examples to time dependent notebook
* DOCS: simple trace notebook

### Changed

* BREAKING: to prototype trace functionality. config name -> class breaks with v0.5.0

### Fixed

* THINNING: patched compatibility of thinning algorithm to work with numpy >= v2. `np.Inf` -> `np.inf`

## [v0.5.0](https://github.com/TomMonks/sim-tools/releases/tag/v0.5.0)  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12204481.svg)](https://doi.org/10.5281/zenodo.12204481)

### Added

* EXPERIMENTAL: added `trace` module with `Traceable` class for colour coding output from different processes and tracking individual patients.

### Fixed

* DIST: fix to `NSPPThinning` sampling to pre-calcualte mean IAT to ensure that correct exponential mean is used.
* DIST: normal distribution allows minimum value and truncates automaticalled instead of resampling.

## [v0.4.0](https://github.com/TomMonks/sim-tools/releases/tag/v0.4.0) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10987685.svg)](https://doi.org/10.5281/zenodo.10987685)

### Changed

* BUILD: Dropped legacy `setuptools` and migrated package build to `hatch`
* BUILD: Removed `setup.py`, `requirements.txt` and `MANIFEST` in favour of `pyproject.toml`

## [v0.3.3](https://github.com/TomMonks/sim-tools/releases/tag/v0.3.3) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10629861.svg)](https://doi.org/10.5281/zenodo.10629861)

### Fixed

* PATCH: `distributions.Discrete` was not returning numpy arrays.

## [v0.3.2](https://github.com/TomMonks/sim-tools/releases/tag/v0.3.2) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10625581.svg)](https://doi.org/10.5281/zenodo.10625581)

## Changed 

* Update Github action to publish to pypi. Use setuptools instead of build

## [v0.3.1](https://github.com/TomMonks/sim-tools/releases/tag/v0.3.1) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10625470.svg)](https://doi.org/10.5281/zenodo.10625470)

### Fixed:

* PYPI has deprecated username and password. PYPI Publish Github action no works with API Token

## [v0.3.0](https://github.com/TomMonks/sim-tools/releases/tag/v0.3.0)

### Added

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

## [v0.2.1](https://github.com/TomMonks/sim-tools/releases/tag/v0.2.1)

### Fixed

* Modified Setup tools to avoid numpy import error on build.
* Updated github action to use up to date actions.

## v0.2.0 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10201726.svg)](https://doi.org/10.5281/zenodo.10201726)

### Added

* Added `sim_tools.distribution` module.  This contains classes representing popular sampling distributions for Discrete-event simulation. All classes encapsulate a `numpy.random.Generator` object, a random seed, and the parameters of a sampling distribution.  

### Changed

* Python has been updated, tested, and patched for 3.10 and 3.11 as well as numpy 1.20+
* Minor linting and code formatting improvement.