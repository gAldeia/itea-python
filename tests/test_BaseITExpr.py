import pytest

import numpy as np

from itea._base import BaseITExpr

from sklearn.base import clone


@pytest.fixture
def linear_baseITExpr():
    r"""Returns an valid instance of the BaseITexpr class. The returned 
    itexpr corresponds to a linear combination of 5 features, with arbitrary
    labels that can be used to test particularities when printing the
    expression.

    The instance is not fitted. All arguments should be named.

    The gradient of this expression should be:
        (1, 1, 1, 1, 1),
    for all\mathbf{x} \in \mathbb{R}^n.
    """

    return BaseITExpr(
        expr = [
            ('id', [1, 0, 0, 0, 0]),
            ('id', [0, 1, 0, 0, 0]),
            ('id', [0, 0, 1, 0, 0]),
            ('id', [0, 0, 0, 1, 0]),
            ('id', [0, 0, 0, 0, 1]),
        ],
        tfuncs = {'id' : lambda x: x},
        labels = ['v1', 'v2', 'under_line', r'\alpha', 'x'],
    )


@pytest.fixture
def nonlinear_baseITExpr():
    r"""
    The instance is not fitted. All arguments should be named.

    The gradient of this expression should be:
        (2*x_1*(1 + x_3), 2*x_2 + x_4, 3*x_3^2, x_0^2, x_1),
    for all\mathbf{x} \in \mathbb{R}^n.
    """

    return BaseITExpr(
        expr = [
            ('id', [2, 0, 0, 0, 0]),
            ('id', [0, 2, 0, 0, 0]),
            ('id', [0, 0, 3, 0, 0]),
            ('id', [2, 0, 0, 1, 0]),
            ('id', [0, 1, 0, 0, 1]),
        ],
        tfuncs = {'id' : lambda x: x},
        labels = ['v1', 'v2', 'under_line', r'\alpha', 'x'],
    )

    
@pytest.fixture
def sample_data():
    """Sample data to test the it expression that makes a linear combination
    of the features.
    """

    X = [[1., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0.],
         [0., 0., 1., 0., 0.],
         [0., 0., 0., 1., 0.],
         [0., 0., 0., 0., 1.],
         [0., 0., 0., 0., 1.]]
    
    # expected output if the model was a linear combination of the variables
    y = [1, 1, 1, 1, 1]

    return X, y


def test_linear_BaseITExpr_to_str(linear_baseITExpr):

    assert linear_baseITExpr.to_str() == (
        r"1.0*id(v1) + 1.0*id(v2) + 1.0*id(under_line) "
        r"+ 1.0*id(\alpha) + 1.0*id(x) + 0.0"
    )


def test_linear_BaseITExpr__str__(linear_baseITExpr):

    assert str(linear_baseITExpr) == linear_baseITExpr.to_str()


def test_linear_BaseITExpr_eval(linear_baseITExpr, sample_data):

    X, _ = sample_data

    # it should be the same matrix, since _eval returns a matrix where each
    # column is one it term, and the it expression is a linear combination
    # of the features, using onli the identity function as transformation func
    assert np.array_equal(linear_baseITExpr._eval(X), X) 


def test_linear_BaseITExpr_complexity(linear_baseITExpr):
    # calculated by hand, the expected complexity is 26, the number of nodes
    # needed to recreate the it expression as a symbolic expression

    assert linear_baseITExpr.complexity() == 26


def test_linear_BaseITExpr_n_terms(linear_baseITExpr):
    assert linear_baseITExpr.n_terms == 5


@pytest.mark.parametrize("logit", [False, True])
def test_linear_BaseITExpr_gradient(linear_baseITExpr, sample_data, logit):

    X, _ = sample_data

    # will be a 3d array
    expected_gradients = np.array(
        [[[1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.],
          [1., 1., 1., 1., 1.]]]
    )

    # derivative of a expit function that uses the itexpr as linear model
    if logit:
        predictions = np.dot(
            linear_baseITExpr._eval(X),
            np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]).T
        )
        
        for c_idx in range(expected_gradients.shape[0]):
            for x_idx in range(expected_gradients.shape[2]): # number of variables
                expected_gradients[c_idx, :, x_idx] = np.divide(
                    np.exp(predictions)[:, c_idx] * expected_gradients[c_idx, :, x_idx],
                    np.power(np.exp(predictions)[:, c_idx] + 1.0, 2)
                )

    assert np.array_equal(
        linear_baseITExpr.gradient(
            X,
            tfuncs_dx = {'id': lambda x: np.ones_like(x)},
            logit = logit
        ),
        expected_gradients
    )
    

@pytest.mark.parametrize("logit", [False, True])
def test_nonlinear_BaseITExpr_gradient(
    nonlinear_baseITExpr, sample_data, logit):

    X, _ = sample_data

    expected_gradients = np.array(
        [[[2., 0., 0., 1., 0.],
          [0., 2., 0., 0., 1.],
          [0., 0., 3., 0., 0.],
          [0., 0., 0., 0., 0.],
          [0., 1., 0., 0., 0.],
          [0., 1., 0., 0., 0.]]]
    )

    # derivative of a expit function that uses the itexpr as linear model
    if logit:
        predictions = np.dot(
            nonlinear_baseITExpr._eval(X),
            np.array([[1.0, 1.0, 1.0, 1.0, 1.0]]).T
        )
        
        for c_idx in range(expected_gradients.shape[0]):
            for x_idx in range(expected_gradients.shape[2]): # number of variables
                expected_gradients[c_idx, :, x_idx] = np.divide(
                    np.exp(predictions)[:, c_idx] * expected_gradients[c_idx, :, x_idx],
                    np.power(np.exp(predictions)[:, c_idx] + 1.0, 2)
                )

    assert np.array_equal(
        nonlinear_baseITExpr.gradient(
            X,
            tfuncs_dx = {'id': lambda x: np.ones_like(x)},
            logit = logit
        ),
        expected_gradients
    )


def test_linear_BaseITExpr_clone(linear_baseITExpr):

    # Create a non-fitted copy of the original scikit regressor
    linear_baseITExpr_copy = clone(linear_baseITExpr)

    assert linear_baseITExpr.get_params() == linear_baseITExpr_copy.get_params()

    # Changing the clone shound't affect the original one
    linear_baseITExpr_copy.expr = [('id', [1, 1, 1])]
    assert linear_baseITExpr.get_params() != linear_baseITExpr_copy.get_params()


def test_linear_BaseITExpr_virtual_methods(linear_baseITExpr, sample_data):

    X, y = sample_data

    with pytest.raises(NotImplementedError):
        linear_baseITExpr.covariance_matrix(X, y)

    with pytest.raises(NotImplementedError):
        linear_baseITExpr.fit(X, y)

    with pytest.raises(NotImplementedError):
        linear_baseITExpr.predict(X)

    with pytest.raises(NotImplementedError):
        linear_baseITExpr.predict_proba(X)