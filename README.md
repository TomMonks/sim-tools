# `sim-tools`: tools to support the Discrete-Event Simulation process in python.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TomMonks/sim-tools/HEAD)
[![DOI](https://zenodo.org/badge/225608065.svg)](https://zenodo.org/badge/latestdoi/225608065)
[![PyPI version fury.io](https://badge.fury.io/py/sim-tools.svg)](https://pypi.python.org/pypi/sim-tools/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/sim-tools/badges/version.svg)](https://anaconda.org/conda-forge/sim-tools)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/sim-tools/badges/platforms.svg)](https://anaconda.org/conda-forge/sim-tools)
[![Read the Docs](https://readthedocs.org/projects/pip/badge/?version=latest)](https://tommonks.github.io/sim-tools)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-360+/)
[![License: MIT](https://img.shields.io/badge/ORCID-0000--0003--2631--4481-brightgreen)](https://orcid.org/0000-0003-2631-4481)


`sim-tools` is being developed to support Discrete-Event Simulation (DES) education and applied simulation research.  It is MIT licensed and freely available to practitioners, students and researchers via [PyPi](https://pypi.org/project/sim-tools/) and [conda-forge](https://anaconda.org/conda-forge/sim-tools)

 # Vision for sim-tools

 1. Deliver high quality reliable code for DES education and practice with full documentation.
 2. Provide a simple to use pythonic interface.
 3. To improve the quality of DES education using FOSS tools and encourage the use of best practice.

# Features:

1. Implementation of classic Optimisation via Simulation procedures such as KN, KN++, OBCA and OBCA-m
2. Distributions module that includes classes that encapsulate a random number stream, seed, and distribution parameters.
3. Implementation of Thinning to sample from Non-stationary poisson processes in a DES.

## Installation

### Pip and PyPi

```bash
pip install sim-tools
```

### Conda-forge

```bash
conda install -c conda-forge sim-tools
```

### Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TomMonks/sim-tools/HEAD)


## Learn how to use `sim-tools`

* Online documentation: https://tommonks.github.io/sim-tools
* Introduction to DES in python: https://health-data-science-or.github.io/simpy-streamlit-tutorial/

## Citation

If you use sim0tools for research, a practical report, education or any reason please include the following citation.

> Monks, Thomas. (2021). sim-tools: tools to support the forecasting process in python. Zenodo. http://doi.org/10.5281/zenodo.4553642

```tex
@software{sim_tools,
  author       = {Thomas Monks},
  title        = {sim-tools: fundamental tools to support the simulation process in python},
  year         = {2021},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.4553642},
  url          = {http://doi.org/10.5281/zenodo.4553642}
}
```

# Online Tutorials

* Optimisation Via Simulation [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TomMonks/sim-tools/blob/master/examples/sw21_tutorial.ipynb)


## Contributing to sim-tools

Please fork Dev, make your modifications, run the unit tests and submit a pull request for review.

Development environment:

* `conda env create -f binder/environment.yml`

* `conda activate sim_tools`

**All contributions are welcome!**
