# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 05-31-2021 by Guilherme Aldeia


"""ITExpr sub-class, specialized to classification task.
"""


import numpy as np

from sklearn.base             import ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import accuracy_score

from itea._base import BaseITExpr


class ITExpr_classifier(BaseITExpr, ClassifierMixin):
    """ITExpr for the classification task. This will be the class 
    in ``ITEA_classifier.bestsol_``.
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

        labels : list of strings, default=[]
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
            fitness (accuracy_score) of the expression on the training data.

        intercept_ : numpy.array of shape (n_classes, )
            intercept array used in the probability estimation for each class
            of the training data.

        coef_ : numpy.array of shape (n_classes, n_terms)
            coefficients used in the probability estimation for each
            class of the training data.

        classes_ : array of shape (n_classes, )
            target classes infered from the training y target data.
        """

        super(ITExpr_classifier, self).__init__(expr=expr, tfuncs=tfuncs, labels=labels)

        self.fit_model = LogisticRegression
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

        covars = np.zeros( (len(self.classes_), self.n_terms+1, self.n_terms+1) )
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
            ``intercept_`` and ``classes_`` will be available.
        """

        if not self._is_fitted:
        
            Z = self._eval(X)

            if np.isfinite(Z).all() and np.all(np.abs(Z) < 1e+100):
                
                self.fit_model_ = self.fit_model()
                
                pred  = self.fit_model_.fit(Z, y).predict(Z)

                self.classes_   = self.fit_model_.classes_.tolist()
                self.intercept_ = self.fit_model_.intercept_.tolist()
                self._fitness   = self.fitness_f(pred, y)
                self.coef_      = self.fit_model_.coef_.tolist()
            else:
                self.classes_   = np.unique(y).tolist()
                self.intercept_ = np.zeros( (len(self.classes_)) ).tolist()
                self._fitness   = np.nan
                self.coef_      = np.ones(
                    (len(self.classes_), self.n_terms) ).tolist()

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
        """

        check_is_fitted(self)
        
        assert self._is_fitted, \
            ("The expression was simplified and has not refitted.")

        assert hasattr(self, 'fit_model_'), \
            ("The expression failed in evaluating finite values during the "
             "training process.")

        X = check_array(X)

        return self.fit_model_.predict_proba(self._eval(X))