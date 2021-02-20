import setuptools
from ovs import __version__

# Read in the requirements.txt file
with open("requirements.txt") as f:
    requirements = []
    for library in f.read().splitlines():
        requirements.append(library)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ovs_py",
    #there must be an way to auto tick up the version number...
    version=__version__,
    author="Thomas Monks",
    author_email="t.m.w.monks@exeter.ac.uk",
    license="The MIT License (MIT)",
    description="Optimisation Via Simulation in PYthon.",
    #read in from readme.md and will appear on PyPi
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TomMonks/ovs-tutorial",
    packages=setuptools.find_packages(),
    #if true look in MANIFEST.in for data files to include
    include_package_data=False,
    #2nd approach to include data is include_package_data=False
    #package_data={"test_package": ["data/*.csv"]},
    #these are for documentation 
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.9',
    install_requires=requirements,
)
