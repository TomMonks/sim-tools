# sim-tools: tools to support the simulation process in python.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TomMonks/sim-tools/HEAD)
[![DOI](https://zenodo.org/badge/225608065.svg)](https://zenodo.org/badge/latestdoi/225608065)
[![PyPI version fury.io](https://badge.fury.io/py/sim-tools.svg)](https://pypi.python.org/pypi/sim-tools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360+/)

sim-tools is being developed to support simulation education and applied simulation research.  It is MIT licensed and freely available to practitioners, students and researchers via PyPi.  There is a longer term plan to make sim-tools available via conda-forge.

 # Vision for sim-tools

 1. Deliver high quality reliable code for simulation education and practice with full documentation.
 2. Provide a simple to use pythonic interface.
 3. To improve the quality of simulation education and encourage the use of best practice.

# Features:

1. Implementation of classic optimisation via Simulation procedures such as KN, KN++, OBCA and OBCA-m

## Two simple ways to explore sim-tools

1. `pip install sim-tools`
2. Click on the launch-binder at the top of this readme. This will open example Jupyter notebooks in the cloud via Binder.

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

# Examples

* Optimisation Via Simulation [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TomMonks/sim-tools/blob/master/examples/sw21_tutorial.ipynb)


## Contributing to sim-tools

Please fork Dev, make your modifications, run the unit tests and submit a pull request for review.

Development environment:

* `conda env create -f binder/environment.yml`

* `conda activate sim_tools_dev`

**All contributions are welcome!**
