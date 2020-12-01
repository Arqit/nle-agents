# TODO provide an apt description to all aspects of the project


# Getting started

Starting with NLE environments is extremely simple, provided one is familiar
with other gym / RL environments.


## Installation

NLE requires `python>=3.5`, `cmake>=3.14` to be installed and available both when building the
package, and at runtime.

On **MacOS**, one can use `Homebrew` as follows:

``` bash
$ brew install cmake
```

On a plain **Ubuntu 18.04** distribution, `cmake` and other dependencies
can be installed by doing:

```bash
# Python and most build deps
$ sudo apt-get install -y build-essential autoconf libtool pkg-config \
    python3-dev python3-pip python3-numpy git flex bison libbz2-dev

# recent cmake version
$ wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
$ sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
$ sudo apt-get update && apt-get --allow-unauthenticated install -y \
    cmake \
    kitware-archive-keyring
```

Afterwards it's a matter of setting up your environment. We advise using a conda
environment for this:

```bash
$ conda create -n nle python=3.8
$ conda activate nle
$ pip install nle
```


NOTE: If you want to extend / develop NLE, please install the package as follows:

``` bash
$ git clone https://github.com/facebookresearch/nle --recursive
$ pip install -e ".[dev]"
$ pre-commit install
```

## To run agent 1

## To run agent 2
