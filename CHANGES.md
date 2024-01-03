# Change log

## v0.3.0

* Distributions classes now have python type hints
* Added distributions example notebook.
* Added `datasets` module and function to load example NSPP dataset.
* Distributions added
    * Erlang (mean and stdev parameters)
    * Beta
    * Gamma
    * Weibull
    * PearsonV
    * PearsonVI
    * Discrete
    * ContinuousEmpirical (linear interpolation between groups)
    * RawEmpirical (resample with replacement from individual X's)
    * TruncatedDistribution (arbitrary truncation of any distribution)



## v0.2.0

* Added `sim_tools.distribution` module.  This contains classes representing popular sampling distributions for Discrete-event simulation. All classes encapsulate a `numpy.random.Generator` object, a random seed, and the parameters of a sampling distribution.  

* Python has been updated, tested, and patched for 3.10 and 3.11 as well as numpy 1.20+

* Minor linting and code formatting improvement.