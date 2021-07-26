# itea-python

![code coverage](https://galdeia.github.io/itea-python/_images/coverage.svg)
![python version](https://galdeia.github.io/itea-python/_images/pythonversion.svg)

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

ITEA is currently in tests and is available
[at test.pypi](https://test.pypi.org/project/itea)

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

## Using the ITEA algorithm

Before jumping into using the library, here are some tips and examples
that might make it easier to use.

ITEA can be used for regression (with the ``ITEA_regressor`` class) or
for classification (``ITEA_classifier``). Both classes inherit from base
classes in the scikit-learn library, and implement very similar methods. 
The simplest use is to create an instance and use the fit method.

```python
from sklearn import datasets

# Loading regression data and fitting an ITEA regressor
from itea.regression import ITEA_regressor

housing_data = datasets.fetch_california_housing()
X_reg, y_reg = housing_data['data'], housing_data['target']
labels       = housing_data['feature_names']

reg = ITEA_regressor(labels=labels).fit(X_reg, y_reg)

# loading classification data and fitting an ITEA classifier
from itea.classification import ITEA_classifier

iris_data      = datasets.load_iris()
X_clf, targets = iris_data['data'], iris_data['target']

clf = ITEA_classifier().fit(X_clf, targets)
```

The convention is to always pass arguments by name for all that are not
mandatory and have default values ​​defined. To specify a setting other than
the default, we must pass the arguments by name:

```python
reg = ITEA_regressor(gens=50, popsize=200).fit(X_reg, y_reg)

# the line below does not work
reg = ITEA_regressor(50, 200).fit(X_reg, y_reg)
```

The [documentation](https://galdeia.github.io/itea-python/index.html) presents
the default values ​​for each algorithm, with exaplanations of what they
represent.

After performing the evolution (fitting the ``ITEA``), the best symbolic
expression can be accessed by the ``bestsol_`` attribute. The best expression
is an already fitted sckit estimator. The bestsol_ is used to predict,
calculate the score, print the expression, and to obtain interpretability
with model-agnostic (or the model-specific ``ITExpr_explainer``) explainers.

The ``ITEA`` instance implements the predict method, but essentially it just
uses bestsol's predict.

```python
final_itexpr = reg.bestsol_

# Will print the expression as string. 
print(final_itexpr)
>>>  9.924*log(MedInc^2 * AveBedrms * Longitude^2) +
7.982*log(MedInc * HouseAge * AveRooms * AveOccup^2 * Longitude^2) +
-9.092*log(HouseAge * AveRooms * AveBedrms * AveOccup^2 * Latitude * Longitude^2) +
0.702*log(HouseAge^2 * AveBedrms * AveOccup^2 * Latitude^2 * Longitude^2) +
-25.846*log(MedInc) +
-62.377

# Returns the predictions for every observation
final_itexpr.predict(X_reg)

# yields the same result as the previous line
reg.predict(X_reg)
```

The ITEA Package also implements some classes focused on interpretability,
providing mechanisms to inspect and better understand the returned symbolic
expressions. We can obtain importance values ​​from expression attributes and
even generate graphs:

```python
explainer = ITExpr_explainer(
    itexpr=final_itexpr, tfuncs=reg.tfuncs).fit(X_reg, y_reg)

explainer.plot_feature_importances(
    X=X_reg,
    importance_method='pe',
    grouping_threshold=0.0,
    barh_kw={'color':'green'}
)
```

![feature importances plot](https://galdeia.github.io/itea-python/_images/_regression_example_17_0.png)

Explainers do not inherit any scikit interfaces, but implements a similar usage.
So, the steps to use the explainers are: 1. Instanciate the explainer; 2. Fit;
3. Generate the plots.

That said, if you're familiar with scikit's ecosystem of regressors and
classifiers, you'll have no problem using ITEA and its explainers.

For more examples, see:

* [A working notebook using ``ITEA_classifier``](https://galdeia.github.io/itea-python/_multiclass_example.html)
* [A working notebook using ``ITEA_regressor``](https://galdeia.github.io/itea-python/_regression_example.html)
* [More examples of the ITEA package](https://galdeia.github.io/itea-python/index.html)
* 

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