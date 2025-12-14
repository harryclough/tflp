
## About

This is the source code used to run the experiments for the paper *Strategic Agent-Based Equilibrium Models for Urban Mobility: the Traffic Filter Location Problem*, which has been accepted to the 26th International Conference on Principles and Practice of Multi-Agent Systems (PRIMA 2025).

Paper Authors: Harry Clough, Gennaro Auricchio, Jie Zhang
Code Author: Harry Clough


## Codebase

The project is written in Python `3.10.6`.

Dependancies are listed in `requirements.txt`. If using pip, they can be installed by running the command `pip install -r requirements.txt` from inside a virtual environment.


### Notebooks

The top level Jupyter notebooks set up and run the experiments.

[`var_param_runner`](./var_param_runner.ipynb) is responsible for running experiments using different configurations of parameters. It is currently configured to use the synthetic networks provided in [`graph.py`](./src/graph.py).

[`dataset_runner`](./dataset_runner.ipynb) load in a parsed network from the Transportation Networks dataset (see below), and runs the edge removal brute force experiment on it.


### Implementation

`/src/` contains the following top-level files, implementing the simulation and equilibrium solvers:

| File | Description |
| ----------- | ----------- |
| [`graph.py`](./src/graph.py) | Implements a `Graph` network with helper methods to e.g. find the shortest path between two points. |
| [`instance.py`](./src/instance.py) | The `Instance` class implements the mathematical model of a transport network described in the paper. Pre-generates all possible paths, making it only useful on small networks. Used by `scipysolver`. |
| [`pathinstance.py`](./src/pathinstance.py) | A subclass of `Instance` that restricts the pool of active paths between each origin destination pair. Used by `colgensolver`. |
| [`colgensolver.py`](./src/colgensolver.py) | Implements a column generation / path swapping algorithm, which is used to find the user equilibrium for a given `PathInstance`. Directly adapted from algorithm proposed by [Lu et al](https://doi.org/10.1016/j.trb.2008.07.005). |
| [`scipysolver.py`](./src/scipysolver.py) | *Deprecated*. Uses a generic mathematical optimisation tool (`scipy.optimize`) to find the user equilibrium for a given `Instance`. |

`/src/experiments/` implements the experiment runners and helper classes, such as the objective functions.

### Dataset

The [`/networks/`](./networks/) folder contains the Anaheim network from the [Transportation Networks](https://github.com/bstabler/TransportationNetworks) GitHub repo. The networks must be parsed using the `parsing networks in Python` script before they can be loaded and run by `dataset_runner`.

## Future Work

The dataset networks can be very large, and so runtimes can be hours to days. Future work would benefit from a re-implementation of the simulation and equilbrium finding code in a more perfomant language like C++ or Rust.

The authors are actively working on a journal extension to this paper, which will included more extensive experimentation, among others.
