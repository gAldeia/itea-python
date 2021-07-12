# itea-python

![code coverage](https://galdeia.github.io/itea-python/_images/coverage.svg)

itea is a python implementation of the Interaction-Transformation Evolutionary
Algorithm described in the paper "Franca, F., & Aldeia, G. (2020).
Interaction-Transformation Evolutionary Algorithm for Symbolic Regression.
Evolutionary Computation, 1-25."

The Interaction-Transformation (IT) representation is a step towards obtaining
simpler and more interpretable results, searching in the mathematical
equations space by means of an evolutionary strategy.

Together with ITEA for Classification and Regression, we provide a
model-specific explainer based on the Partial Effects to help users get a
better understanding of the resulting expressions.

This implementation is based on the scikit-learn package and the implementations
of the estimators follow their guidelines.


## Documentation

Documentation is available [here](https://galdeia.github.io/itea-python/).


## Installation

ITEA is currently in tests and is available [at test.pypi](https://test.pypi.org/project/itea)

Since packages uploaded on test.pypi index will search for dependencies in 
the same index, to install the test version you can run:

```shell
$ pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple itea
```

Alternatively, you can download the source code at the 
[itea-python GitHub page](https://github.com/gAldeia/itea-python) and run:

```shell
$ pip install .
```

## Testing

To run the test suite, on the root of the project, call:

```shell
$ python3 setup.py pytest
```


## Profiling

A simple execution of the ``ITEA_classifier`` and ``ITEA_regressor`` using a
toy data set is implemented inside the folder ``./profiling/``. This is intended
to test and report the time each function took, and it is used when optimizing
the package.

To properly run the profiles, you need to install via pip the snakeviz package,
which will be used to generate useful plots of the function calls and execution
times.

To run the profiling, on the root of the project, call:

```shell
$ make profile
```

This rule of the Makefile is not executed when make is called, and it is only
to simplify the process of executing multiple profiling tests.


## Benchmarking

The ITEA algorithm was proposed in _"Franca, F., & Aldeia, G. (2020).
Interaction-Transformation Evolutionary Algorithm for Symbolic Regression.
Evolutionary Computation, 1-25."_.

The original paper evaluated the algorithm for the regression problem on
several popular data sets.

To ensure this implementation is aligned with the results of the original 
paper, the ``./benchmark/regression`` folder has a script to run the algorithm
with approximately the same configuration for the original paper, and save
the results in a .csv file.

To run the benchmark, inside the ``./benchmark/regression`` folder:

```shell
$ python regression_benchmark.py <data set name>
```

You can run multiple processes at the same time, as long as there is no
concurrent execution over the same data set. The .lock file will prevent
racing conditions.

You can check the performance of the results by analyzing the 
``regression_benchmark_res.csv``. For a quick check, on python:

```python
import pandas as pd

results = pd.read_csv('regression_benchmark_res.csv')

# Will print the mean of all executions in the results file for each data set
print(results.drop(columns=['Rep']).groupby('Dataset').mean())
```


## Contributing

This is still in active development. Feel free to contact the developers with
suggestions, critics, or questions. You can always raise an issue on GitHub!