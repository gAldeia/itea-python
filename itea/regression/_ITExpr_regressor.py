# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 05-31-2021 by Guilherme Aldeia


"""ITExpr sub-class, specialized to regression task.
"""


import numpy as np

from sklearn.base             import RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.linear_model     import LinearRegression
from sklearn.metrics          import mean_squared_error

from itea._base import BaseITExpr


class ITExpr_regressor(BaseITExpr, RegressorMixin):
    """ITExpr for the regression task. This will be the class 
    in ``ITEA_regressor.bestsol_``.
    """

    def __init__(self, *, expr, tfuncs, labels = [], **kwargs):
        r"""Constructor method.

        Parameters
        ----------
        expr : list of Tuple[Transformation, Interaction]
            list if IT terms to create an IT expression.
        tfuncs : dict
            should always be a dict where the
            keys are the names of the transformation functions and 
            the values are unary vectorized functions (for example,
            numpy functions). For user-defined functions, see
            numpy.vectorize for more informations on how to vectorize
            your transformation functions.

        labels : list, default=[]
            list containing the labels of the variables that will be used.
            When the list of labels is empty, the variables are named
            :math:`x_0, x_1, \cdots`.

        Attributes
        ----------
        n_terms : int
            number of infered IT terms.

        is_fitted : bool
            boolean variable indicating if the ITExpr was fitted before.

        _fitness : float
            fitness (RMSE) of the expression on the training data.

        intercept_ : float 
            regression intercept.

        coef_ : numpy.array of shape (n_terms, )
            coefficients for each term.
        """

        super(ITExpr_regressor, self).__init__(
            expr=expr, tfuncs=tfuncs, labels=labels)

        self.fit_model = LinearRegression
        self.fitness_f = lambda pred, y: mean_squared_error(
            pred, y, squared=False)


    def covariance_matrix(self, X, y):
        """Estimation of the covariance matrix of the coefficients.

        Parameters
        ----------
        X: numpy.array of shape (n_samples, n_features)

        Returns
        -------
        covar : numpy.array of shape (n_terms+1, n_terms+1)
            covariance matrix of the coefficients.

            The last row/column is the intercept.
        """

        N = X.shape[0]
        p = self.n_terms + 1

        residuals = y - self.predict(X)
        
        residual_sum_of_squares = residuals.T @ residuals

        sigma_squared_hat = residual_sum_of_squares / (N - p)
        
        X_design = np.ones( (N, p) )
        X_design[:, :-1] = self._eval(X)

        try:
            return np.linalg.inv(X_design.T @ X_design) * sigma_squared_hat
        except np.linalg.LinAlgError as e:
            return np.linalg.pinv(X_design.T @ X_design) * sigma_squared_hat


    def fit(self, X, y):
        """Fits the linear model created by combining the IT terms.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            training data.

        y : array-like of shape (n_samples, )
            expected values. 

        Returns
        -------
        self : ITExpr_regressor
            itexpr after fitting the coefficients and intercept.
            Only after fitting the model that the attributes ``coef_`` and
            ``intercept_`` will be available.
        """
                
        if not self._is_fitted:
        
            Z = self._eval(X)

            if np.isfinite(Z).all() and np.all(np.abs(Z) < 1e+100):
                self.fit_model_ = self.fit_model()

                pred  = self.fit_model_.fit(Z, y).predict(Z)

                self.coef_      = self.fit_model_.coef_.tolist()
                self.intercept_ = self.fit_model_.intercept_
                self._fitness   = self.fitness_f(pred, y)
            else:
                self.coef_      = np.ones(self.n_terms)
                self.intercept_ = 0.0
                self._fitness   = np.nan

            self._is_fitted = True

        return self


    def predict(self, X):
        """Predicts the response value for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            samples to be predicted. Must be a two-dimensional array.

        Returns
        -------
        p : numpy.array of shape (n_samples, )
            predicted response value for each sample.
        """

        check_is_fitted(self)

        assert self._is_fitted, \
            ("The expression was simplified and has not refitted.")

        X = check_array(X)

        return np.dot(self._eval(X), self.coef_) + self.intercept_