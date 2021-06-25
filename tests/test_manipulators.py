import numpy as np

from itea._manipulators.generator  import uniform
from itea._manipulators.sanitizer  import sanitize
from itea._manipulators.simplifier import simplify_by_coef, simplify_by_var

from itea.regression     import ITExpr_regressor
from itea.classification import ITExpr_classifier


# Setting a simple configuration to evaluate the methods
max_terms = 5
expolim  = (-2, 2)
tfuncs   = {'id': lambda x: x, 'sin': np.sin}
nvars    = 5


def test_uniform_generator_valid_exprs():

    generator = uniform(
        max_terms, expolim, tfuncs, nvars, np.random.RandomState(42))
    
    population = [next(generator) for _ in range(100)]

    for p in population:
        assert type(p) == list

        assert 1 <= len(p) <= max_terms

        for f, t in p:
            assert f in tfuncs.keys()
            assert all(ti <= expolim[1] for ti in t)
            assert all(ti >= expolim[0] for ti in t)


def test_uniform_generator_random_state():

    generator = uniform(
        max_terms, expolim, tfuncs, nvars, np.random.RandomState(42))
    
    population = [next(generator) for _ in range(100)]

    generator2  = uniform(
        max_terms, expolim, tfuncs, nvars, np.random.RandomState(42))
    
    population2 = [next(generator2) for _ in range(100)]

    for p1, p2 in zip(population, population2):
        for (f1, t1), (f2, t2) in zip(p1, p2):
            assert f1 == f2
            assert t1 == t2


def test_sanitizer():
    expr = [
        ('id',  [0, 1, 0]),
        ('log', [0, 1, 0]),
        ('id',  [0, 1, 0]),
    ]

    clean_expr = sanitize(expr)

    assert len(clean_expr) == len(expr) - 1
    assert ('id',  [0, 1, 0]) in clean_expr 
    assert ('log', [0, 1, 0]) in clean_expr


def test_simplifier_coef_ITExpr_regressor():
    itexpr_regressor = ITExpr_regressor(
        expr=[
            ('log', [0, 1, 0]),
            ('id',  [0, 1, 0]),
        ],
        tfuncs = {'id': lambda x : x, 'log': np.log}
    )

    # sample coefs to test
    itexpr_regressor.coef_ = np.array([1e-7, 1.0])

    itexpr_regressor_simplified = simplify_by_coef(itexpr=itexpr_regressor)
    
    assert itexpr_regressor_simplified.n_terms == 1


def test_simplifier_coef_ITExpr_regressor():
    itexpr_classifier = ITExpr_classifier(
        expr=[
            ('id',  [1, 0, 0]),
            ('id',  [0, 1, 0]),
            ('id',  [0, 0, 1]),
            ('log', [0, 1, 0]),
        ],
        tfuncs = {'id': lambda x : x, 'log': np.log}
    )

    # If at least one class (one row of coefficients) have a value greater
    # than the default 1e-5, then the term is not discarded.
    itexpr_classifier.coef_ = np.array([ 
        [1e-7, 1e+0, 1e-8, 1e-8],
        [1e-8, 1e-7, 1e+2, 1e-7]
    ])

    # coefficients should be a numpy array

    itexpr_classifier_simplified = simplify_by_coef(itexpr=itexpr_classifier)
    
    # The eliminated terms: [True, False, False, True]
    assert itexpr_classifier_simplified.n_terms == 2


def test_simplifier_var():
    
    # We'll create a constant transformation function. It should be removed.
    itexpr_regressor = ITExpr_regressor(
        expr=[
            ('const', [1, 0, 0]),
            ('id',    [0, 1, 0]),
            ('log',   [0, 1, 1]),
            ('const', [0, 1, 0]),
        ],
        tfuncs = {
            'id': lambda x : x,
            'log': np.log,
            'const': lambda x: np.ones_like(x)
        }
    )

    itexpr_regressor.coef_ = np.array([1e-7, 1e-2, 1.0, 100])

    X = np.array([
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
        [6, 6, 6],
        [7, 7, 7],
    ])

    itexpr_simplified = simplify_by_var(itexpr=itexpr_regressor, X = X)

    assert itexpr_simplified.n_terms == 2