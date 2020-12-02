# Using a DQN and Monte-Carlo Tree Search To Play NetHack
This repository contains 2 reinforcement learning implementations to play the game of NetHack. Nethack is popular single-player, terminal-based, rogue-like game that is procedurally  generated, stochastic, and challenging. The NetHack Learning Environment (NLE) is a reinforcement-learning environment based on NetHack and OpenAI's *Gym* framework and was designed to pose a challenge to the current state-of-the-art algorithms. Due to its unique procedurally-generated nature, this testbed environment encourages advancements in various aspects such as exploration,  planning  and  skill  acquisition amongst many others. We present 2 RL algorithms in this repo, namely, a Deep Q-learning Network and a Monte-Carlo Tree Search (MCTS) algorithm. From these 2 approaches, the MCTS approach consistently achieves superior results (and is therefore the recommended implementation) as opposed to the DQN-based approach.
<p align="center">
  <img src="https://nethackwiki.com/mediawiki/images/b/b4/UnNetHack.png" width="600" height = "300">
</p>

To run the agents on Nethack Learning Environment you need to install NLE.


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
## To run agent 1 (MCTS)
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
![Results](https://github.com/Arqit/nle-agents/tree/OptimizedImplementation/src/agent1) from the runs in the report (for the MCTS implementation)
## To run agent 2 (DQN)
You may download the pre-trained weights [here](https://drive.google.com/file/d/1vXTV7TNSSNkrkDtfwyrJ99hPz5i1511D/view?usp=sharing).<br>
To train the DQN:
```bash
$ python3 src/agent2/train.py
```
To evaluate the model, please specify the seeds you would like to evaluate in `evaluation.py` and execute:
```bash
$ python3 src/agent2/evaluation.py
```
To use the agent, import MyAgent.py, configure the hyper-parameter dictionary and create an agent by:
```bash
hyper_params = {...}
agent = MyAgent(
        env.observation_space,  # assuming that we are taking the world as input
        env.action_space,
        train=True,
        replay_buffer=replay_buffer,
        use_double_dqn=hyper_params['use-double-dqn'],
        lr=hyper_params['learning-rate'],
        batch_size=hyper_params['batch-size'],
        discount_factor=hyper_params['discount-factor'],
        beta=hyper_params['beta'],
        prior_eps=hyper_params['prior_eps']
    )
```
## Proudly developed by:
- Mayur Ranchod (1601745)
- Wesley Earl Stander (1056114)
- Joshua Greyling (1616664)
- Agang Lebethe (1610338)
