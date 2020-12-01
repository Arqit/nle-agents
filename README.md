# Getting started

To run the agents on Nethack Learning Environment you need to install NLE. Agent 1 is a Monte-Carlo Tree Search agent and agent 2 is a Deep Q-Learning agent utilizing a Q-Network.


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
## To run agent 1
To generate a ttyrec and stats.csv:
```bash
$ python3 src/agent1/save_run.py
```
To evaluate a random seed:
```bash
$ python3 src/agent1/evaluation.py
```
To run tests on the five seeds in the paper:
```bash
$ python3 src/agent1/RunTests.py
```
To use the agent import MyAgent.py and Node.py then create an agent by:
```bash
agent = MyAgent(env.observation_space, env.action_space, seeds=env.get_seeds())
```

## To run agent 2
