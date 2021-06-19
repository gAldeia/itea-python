# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.1
# Last modified: 06-18-2021 by Guilherme Aldeia


"""ITExpr sub-class, specialized to classification task.
"""


import numpy as np

from sklearn.base              import ClassifierMixin
from sklearn.utils.validation  import check_array, check_is_fitted
from sklearn.metrics           import accuracy_score
from sklearn.linear_model._sag import sag_solver
from sklearn.preprocessing     import LabelEncoder
from sklearn.utils.extmath     import row_norms, safe_sparse_dot, softmax
from sklearn.exceptions        import NotFittedError


from itea._base import BaseITExpr


class ITExpr_classifier(BaseITExpr, ClassifierMixin):
    """ITExpr for the classification task. This will be the class 
    in ``ITEA_classifier.bestsol_``.
    """

    def __init__(self, *,
        expr, tfuncs, labels = [],
        max_iter=100, alpha=0., beta=0.,
        **kwargs
    ):
        r"""Constructor method.

        Parameters
        ----------
        expr : list of Tuple[Transformation, Interaction]
            list of IT terms to create an IT expression.

        tfuncs : dict
            should always be a dict where the
            keys are the names of the transformation functions and 
            the values are unary vectorized functions (for example,
            numpy functions). For user-defined functions, see
            numpy.vectorize for more information on how to vectorize
            your transformation functions.

        labels : list of strings, default=[]
            list containing the labels of the variables that will be used.
            When the list of labels is empty, the variables are named
            :math:`x_0, x_1, \cdots`.

        max_iter : int, default=100
            the maximum number of iterations that the optimization gradient
            method should perform to adjust the coefficients of the linear
            model used as the decision function in the inner logistic
            regression method implemented in the ``ITExpr_classifier``. 
            Smaller values can improve performance, at the cost of a weaker
            adjustment.

        alpha : float, default = 0.0
            The logistic regressor will use the saga solver with a elastic net
            regularization. Alpha parameter controls the L1 regularization.

        beta : float, default = 0.0
            The logistic regressor will use the saga solver with a elastic net
            regularization. Beta parameter controls the L2 regularization.

        Attributes
        ----------
        n_terms : int
            number of inferred IT terms.

        is_fitted : bool
            boolean variable indicating if the ITExpr was fitted before.

        _fitness : float
            fitness (accuracy_score) of the expression on the training data.

        intercept_ : numpy.array of shape (n_classes, )
            intercept array used in the probability estimation for each class
            of the training data.

        coef_ : numpy.array of shape (n_classes, n_terms)
            coefficients used in the probability estimation for each
            class of the training data.

        classes_ : array of shape (n_classes, )
            target classes inferred from the training y target data.

        Notes
        -----
        The saga is described in the paper:
        "Defazio, A., Bach F. & Lacoste-Julien S. (2014). SAGA: A Fast
        Incremental Gradient Method With Support for Non-Strongly Convex
        Composite Objectives"

        """

        super(ITExpr_classifier, self).__init__(
            expr=expr, tfuncs=tfuncs, labels=labels)

        self.max_iter = max_iter
        self.alpha    = alpha
        self.beta     = beta

        self.fitness_f = accuracy_score


    def covariance_matrix(self, X, y):
        """Estimation of the covariance matrix of the coefficients.

        Parameters
        ----------
        X: numpy.array of shape (n_samples, n_features)

        Returns
        -------
        covar : numpy.array of shape (n_classes, n_terms+1, n_terms+1)
            each element in ``covar`` will be the covariance matrix to the
            logistic regressor when considering the classes as a one vs all 
            problem.

            The last row/column of each ``covar[i]`` is the intercept.
        """

        X_design = np.ones( (X.shape[0], self.n_terms + 1 ) )
        X_design[:, :-1] = self._eval(X)

        probs = self.predict_proba(X)

        covars = np.zeros((len(self.classes_), self.n_terms+1, self.n_terms+1))
        for class_id in range(len(self.classes_)):

            # Estimating as a one vs all classification for each class
            prob_class1 = probs[:, class_id]
            prob_class0 = np.sum(
                probs[:, [i for i in range(len(self.classes_))
                    if i != class_id]],
                axis=1)

            V = np.diagflat(np.product([prob_class0, prob_class1], axis=0))

            try:
                covars[class_id, :, :] = np.linalg.inv(
                    X_design.T @ V @ X_design)
            except np.linalg.LinAlgError as e:
                # singular matrix, lets use the pseudo inverse
                covars[class_id, :, :] = np.linalg.pinv(
                    X_design.T @ V @ X_design)

        return covars


    def fit(self, X, y):
        """Fits the logistic regression with the IT expression as the linear
        method.

        This method performs the transformation of the original data in X to 
        the IT expression domain then fits a logistic regressor using the
        IT expression as decision function. The logistic regressor is fitted
        by means of the saga method without any penalties.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            training data.

        y : array-like of shape (n_samples, )
            target vector. Can be a binary classification problem or a 
            multi-class classification problem.

        Returns
        -------
        self : ITExpr_classifier
            itexpr after fitting the coefficients and intercept.
            Only after fitting the model that the attributes ``coef_``,
            ``intercept_``, and ``classes_`` will be available.

        Notes
        -----
        This fit method does not check if the input is consistent, to minimize
        the overhead since the ``ITEA_classifier`` will work with a population
        of ``ITExpr_classifier`` instances. The input is then checked in 
        the fit method from ``ITEA_classifier``. If you want to use the fit
        method directly from the ``ITExpr_classifier``, it is recommended that
        you do the check with ``check_array` `that scikit-learn provides in
        ``sklearn.utils.validation``.
        """

        if not self._is_fitted:
        
            Z = self._eval(X)

            if np.isfinite(Z).all() and np.all(np.abs(Z) < 1e+200):
                # using the LinearRegression from scikit, the fit should be
                # simple as this:
                # from sklearn.linear_model import LogisticRegression
                # fit_model_ = LogisticRegression(solver='saga', penalty='none')
                # self.coef_      = fit_model_.coef_.tolist()
                # self.classes_   = fit_model_.classes_.tolist()
                # self.intercept_ = fit_model_.intercept_
                # self._fitness   = self.fitness_f(fit_model_.predict(Z), y)
                
                self.classes_  = np.unique(y)
                                
                n_classes = len(self.classes_)
                n_terms   = Z.shape[1]
                
                max_squared_sum = row_norms(Z, squared=True).max()

                if n_classes > 2:
                    multi_class = 'multinomial'
                        
                    # from scikit: "SAG multinomial solver needs LabelEncoder"
                    target = LabelEncoder().fit_transform(y) \
                        .astype(Z.dtype, copy=False)

                    w0 = np.zeros((self.classes_.size, n_terms + 1),
                        order='F', dtype=Z.dtype)
                    
                    warm_start = {'coef': w0.T}
                else:
                    multi_class = 'log'
                    
                    n_classes -= 1
                    
                    target = np.ones(y.shape, dtype=Z.dtype)
                    target[~(y == self.classes_[1])] = -1.
                    
                    w0 = np.zeros(n_terms + 1, dtype=Z.dtype)
                    
                    warm_start = {'coef': np.expand_dims(w0, axis=1)}
                
                coef_, n_iter_, _ = sag_solver(
                    Z, target, sample_weight=None, loss=multi_class,
                    alpha=self.alpha, beta=self.beta, max_iter=self.max_iter,
                    tol=0.001, verbose=0, random_state=42,
                    check_input=False, max_squared_sum=max_squared_sum,
                    warm_start_mem=warm_start, is_saga=True
                )
                
                if n_classes <= 2:
                    coef_ = coef_.reshape(n_classes, n_terms + 1)
                
                intercept_ = coef_[:, -1]
                coef_      = coef_[:, :-1]

                # Saving the fitted parameters   
                self.classes_   = self.classes_.tolist()
                self.intercept_ = intercept_.tolist()
                self.coef_      = coef_.tolist()

                # setting fitted to true to use prediction below
                self._is_fitted = True

                self._fitness   = self.fitness_f(self.predict(X), y)
            else:
                self.classes_   = np.unique(y).tolist()
                self.intercept_ = np.zeros( (len(self.classes_)) ).tolist()
                self._fitness   = -np.inf
                self.coef_      = np.ones(
                    (len(self.classes_), self.n_terms) ).tolist()

                # Failed to fit. Default values were set and the is_fitted
                # is set to true to avoid repeated failing fits.
                self._is_fitted = True

        return self


    def predict(self, X):
        """Predict class target for each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            samples to be predicted. Must be a two-dimensional array.

        Returns
        -------
        p : numpy.array of shape (n_samples, )
            predicted target for each sample.

        Raises
        ------
            NotFittedError
                If the expression was not fitted before calling this method.
        """

        probabilities = self.predict_proba(X)

        return np.array(self.classes_)[np.argmax(probabilities, axis=1)]


    def predict_proba(self, X):
        """Predict probabilities for each possible target for
        each sample in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            samples to be predicted. Must be a two-dimensional array.

        Returns
        -------
        p : numpy.array of shape (n_samples, n_classes)
            prediction probability for each class target for each sample.

        Raises
        ------
            NotFittedError
                If the expression was not fitted before calling this method.
        """

        # scikit check - searches for attributes ending with '_'
        check_is_fitted(self)

        # my check, which indicates if the expression was changed by
        # manipulators or not fitted
        if not self._is_fitted:
            raise NotFittedError(
                "The expression was simplified and has not refitted.")

        X = check_array(X)

        prob = (
            safe_sparse_dot(
                self._eval(X), np.array(self.coef_).T
            ) + np.array(self.intercept_)
        )
        
        # If is a binary classification, then we need to create the
        # complementary probability for the second class
        if len(self.classes_) <= 2:
            prob = np.hstack( (np.ones(X.shape[0]).reshape(-1, 1), prob) )  
            prob[:, 0] -= prob[:, 1]
        
        return softmax(prob)