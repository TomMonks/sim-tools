import setuptools

# patch ed - build error due to imporing numpy into __init__
# from sim_tools import __version__
VERSION = "0.3.2"


# Read in the requirements.txt file
with open("requirements.txt") as f:
    requirements = []
    for library in f.read().splitlines():
        requirements.append(library)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sim-tools",
    version=VERSION,
    author="Thomas Monks",
    author_email="t.m.w.monks@exeter.ac.uk",
    license="The MIT License (MIT)",
    description="Simulation Tools for Education and Practice",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TomMonks/sim-tools",
    packages=setuptools.find_packages(),
    #if true look in MANIFEST.in for data files to include
    include_package_data=False,
    #2nd approach to include data is include_package_data=False
    #package_data={"test_package": ["data/*.csv"]},
    #these are for documentation 
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=requirements,
)
