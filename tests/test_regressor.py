import pytest
from scipy.optimize import check_grad

import numpy           as np
import jax.numpy       as jnp

from itea.regression import ITExpr_regressor, ITEA_regressor

from jax import grad, vmap

from sklearn.datasets     import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.exceptions   import NotFittedError

from sklearn.utils._testing import ignore_warnings

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
    
    The ITExpr has no explicit labels.
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
    """non linear expression.
    
    The ITExpr has no explicit labels."""

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


def test_linear_ITExpr_equals_scikit_linearRegression(
    linear_ITExpr, regression_toy_data):

    X, y, coef = regression_toy_data

    # Fitting the linear model, which will correspond to a linear regression
    itexpr_reg = linear_ITExpr.fit(X, y)
    
    scikit_reg = LinearRegression().fit(X, y)

    # They should give the exact same coefficients and intercept, with same
    # shapes and vales, and even have the score() function with same return val
    assert np.array_equal(itexpr_reg.coef_, scikit_reg.coef_)
    assert np.array_equal(itexpr_reg.intercept_, scikit_reg.intercept_)
    assert np.array_equal(itexpr_reg.score(X, y), scikit_reg.score(X, y))
    


def test_linear_ITExpr_predict(
    linear_ITExpr, regression_toy_data):

    X, y, coef = regression_toy_data

    assert np.allclose(linear_ITExpr.fit(X, y).predict(X), y)


@ignore_warnings(category=RuntimeWarning)
def test_linear_ITExpr_predict_nan():

    # Forcing to have an expression with 1/x
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1/1, 1/2, 1/3, 1/4, 1/5])

    nan_input = np.array([[0]])

    itexpr = ITExpr_regressor(
        expr = [
            ('id', [-1.0]),
        ],
        tfuncs = tfuncs
    )

    # shoudn't raise any error, and should return the intercept
    assert np.allclose(
        itexpr.fit(X, y).predict(nan_input),
        itexpr.intercept_)


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

    expected_cov_params = np.array(
        [[ 2.36204730e+02,  7.30247102e+00, -1.40861044e-01, -6.13965636e+01],
         [ 7.30247102e+00,  3.15815042e+02,  8.11634925e-01, -2.23485814e+02],
         [-1.40861044e-01,  8.11634925e-01,  3.24270976e-01, -4.45872482e-01],
         [-6.13965636e+01, -2.23485814e+02, -4.45872482e-01,  2.24715215e+02]])

    assert np.allclose(
        nonlinear_ITExpr.covariance_matrix(X, y),
        expected_cov_params
    )

    # The 'expected_cov_params' was calculated with statsmodels using python
    # 3.8. The statsmodels depends on a package that is not being updated
    # anymore, so to avoid the crash of this test I've hardcoded the expected
    # result. Using python <3.9 is possible to obtain this very matrix by
    # uncommenting the lines below. It will use statsmodels to create a
    # linear regressor with the transformation functions of the
    # nonlinear_ITExpr and calculate the covariance matrix

    #X_with_intercept = np.ones( (X.shape[0], nonlinear_ITExpr.n_terms + 1))
    #X_with_intercept[:, :-1] = nonlinear_ITExpr._eval(X)
    
    #import statsmodels.api as sm

    #ols = sm.OLS(y, X_with_intercept)
    #ols_result = ols.fit()

    #print(ols_result.cov_params()) # getting the result to use it hardcoded
    
    #assert np.allclose(nonlinear_ITExpr.covariance_matrix(X, y),
    #    ols_result.cov_params())


def test_ITEA_regressor_fit_predict(regression_toy_data):
    X, y, coef = regression_toy_data

    # Passing simple labels, tfuncs and tfuncs_dx to suppress the warnings
    reg = ITEA_regressor(
        gens=10, popsize=10, verbose=2,
        random_state=42,
        labels = [f'x_{i}' for i in range(len(X[0]))],
        tfuncs = tfuncs,
        tfuncs_dx = tfuncs_dx
    ).fit(X, y)

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


def test_one_individual_one_generation(regression_toy_data):
    X, y, coef = regression_toy_data

    # Should have a valid fitted expression after 1 generation.
    # Passing simple labels, tfuncs and tfuncs_dx to suppress the warnings
    reg = ITEA_regressor(
        gens=1, popsize=1, verbose=-1,
        random_state=42,
        labels = [f'x_{i}' for i in range(len(X[0]))],
        tfuncs = tfuncs,
        tfuncs_dx = tfuncs_dx
    ).fit(X, y)

    assert hasattr(reg, 'bestsol_')
    assert hasattr(reg, 'fitness_')
    assert np.isfinite(reg.bestsol_._fitness)

    # NOTE: the algorithm does not guarantee that a one individual populaiton
    # will always have a valid expression after n generaions, because there
    # is no elitism and there is a chance that the tournament ends up selecting
    # two bad expressions to compete.