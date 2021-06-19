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

This implementation is based on scikit-learn package and the implementations
of the estimators follow their guidelines.

## Documentation

Documentation is available [here](https://galdeia.github.io/itea-python/).

## Installation

ITEA is currently in tests and is available [at test.pypi](https://test.pypi.org/project/itea)

Since packages uploaded on test.pypi index will search for dependencies in 
the same index, to install the test version you can run:

> $ pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple itea
