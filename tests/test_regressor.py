import pytest

import numpy           as np
import jax.numpy       as jnp
import statsmodels.api as sm

from itea.regression import ITExpr_regressor, ITEA_regressor

from jax import grad, vmap

from sklearn.datasets   import make_regression
from sklearn.exceptions import NotFittedError


# Using the identity, one trigonometric and one non-linear function
tfuncs = {
    'id'       : lambda x: x,
    'sin'      : jnp.sin,
    'sqrt.abs' : lambda x: jnp.sqrt(jnp.abs(x)), 
}

# analitically calculated derivatives
tfuncs_dx = {
    'sin'      : np.cos,
    'sqrt.abs' : lambda x: x/( 2*(np.abs(x)**(3/2)) ),
    'id'       : lambda x: np.ones_like(x),
}

# Automatic differentiation derivatives
tfuncs_dx_jax = {k : vmap(grad(v)) for k, v in tfuncs.items()}


@pytest.fixture
def regression_toy_data():
    """Toy linear data set to test different regressors.
    
    Returns three values: X, y and the coefs for generating y = coefs * X."""

    return make_regression(
        n_samples     = 100,
        n_features    = 3,
        n_informative = 3,
        random_state  = 0,
        noise         = 0.0,
        bias          = 100.0,
        coef          = True
    )

@pytest.fixture
def linear_ITExpr():
    """Linear IT expresison that should fit perfectly to the toy data set.
    
    The ITExpr has no explicit labels in this case.
    """

    return ITExpr_regressor(
        expr = [
            ('id', [1, 0, 0]),
            ('id', [0, 1, 0]),
            ('id', [0, 0, 1])
        ],
        tfuncs = tfuncs
    )

@pytest.fixture
def nonlinear_ITExpr():
    """non linear expression."""

    return ITExpr_regressor(
        expr = [
            ('sin',      [0,  2, 0]),
            ('sqrt.abs', [1,  0, 1]),
            ('id',       [0, -1, 0]),
        ],
        tfuncs = tfuncs,
    )


def test_initial_state(linear_ITExpr):

    assert linear_ITExpr._is_fitted == False
    assert linear_ITExpr._fitness   == np.inf

    assert not hasattr(linear_ITExpr, 'coef_')
    assert not hasattr(linear_ITExpr, 'intercept_')


def test_linear_ITExpr_evaluation(
    linear_ITExpr, regression_toy_data):

    X, y, coef = regression_toy_data
    
    assert np.allclose(X, linear_ITExpr._eval(X))


def test_linear_ITExpr_fit(
    linear_ITExpr, regression_toy_data):

    X, y, coef = regression_toy_data

    with pytest.raises(NotFittedError):
        linear_ITExpr.predict(X)

    linear_ITExpr.fit(X, y)

    assert np.array(linear_ITExpr.coef_).ndim == 1
    assert np.isfinite(linear_ITExpr.intercept_)

    # Shoudnt raise an error anymore    
    linear_ITExpr.predict(X)

    # The ITExpr is exactly the same original expresison. must have almost
    # perfect results.
    assert np.allclose(linear_ITExpr.coef_, coef)
    assert np.isclose(linear_ITExpr._fitness, 0.0)


def test_linear_ITExpr_predict(
    linear_ITExpr, regression_toy_data):

    X, y, coef = regression_toy_data

    assert np.allclose(linear_ITExpr.fit(X, y).predict(X), y)


def test_nonlinear_ITExpr_derivatives_with_jax(
    nonlinear_ITExpr, regression_toy_data):

    X, y, coef = regression_toy_data

    assert np.allclose(
        nonlinear_ITExpr.gradient(X, tfuncs_dx),
        nonlinear_ITExpr.gradient(X, tfuncs_dx_jax)
    )


def test_nonlinear_ITExpr_covariance_matrix(
    nonlinear_ITExpr, regression_toy_data):

    X, y, coef = regression_toy_data

    nonlinear_ITExpr.fit(X, y)

    # Using statsmodels to create a linear regressor with the transformation
    # functions of the nonlinear_ITExpr and evaluate the covariance matrix
    X_with_intercept = np.ones( (X.shape[0], nonlinear_ITExpr.n_terms + 1) )
    X_with_intercept[:, :-1] = nonlinear_ITExpr._eval(X)

    ols = sm.OLS(y, X_with_intercept)
    ols_result = ols.fit()

    assert np.allclose(
        nonlinear_ITExpr.covariance_matrix(X, y),
        ols_result.cov_params()
    )


def test_ITEA_regressor_fit_predict(regression_toy_data):
    X, y, coef = regression_toy_data

    reg = ITEA_regressor(
        gens=10, popsize=10, verbose=2, random_state=42).fit(X, y)

    # The fitness and bestsol attributes should exist after fit
    assert hasattr(reg, 'bestsol_')
    assert hasattr(reg, 'fitness_')

    # Those atributes should be shared between the ITEA and best ITExpr
    # (fitness is private for ITExpr. It is not created only after fitting
    # the model and has meaning on the evolution context. The ITEA have the 
    # fitness_ attribute for convenience. Idealy (like any other scikit model),
    # the score() is implemented to assess the model performance.)

    assert reg.fitness_ == reg.bestsol_._fitness
    assert np.allclose(reg.score(X, y), reg.bestsol_.score(X, y))

    # predict() called on ITEA or ITExpr always corresponds to calling it
    # directly on the ITExpr
    assert np.allclose(reg.predict(X), reg.bestsol_.predict(X))