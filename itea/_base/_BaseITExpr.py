# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.2
# Last modified: 06-25-2021 by Guilherme Aldeia


"""Base class to represent an IT expression.
"""

import numpy as np

from sklearn.base             import BaseEstimator
from sklearn.utils.validation import check_array


class BaseITExpr(BaseEstimator):
    """This class describes the structure that an ``ITExpr`` should have, and
    implements only the methods that have similar behavior for classification
    and regression.

    The ITEA implementations for classification and regression will create
    a population of ``ITExpr`` instances and evolve this population to find a
    final best solution ``itea.bestsol_``.

    The best solution will be a scikit estimator and can be used in many scikit
    methods. It can also be used to create ICE and PDP plots, which are
    particularly interesting to complement explanations given by the
    ``ITExpr_explainer``.

    Methods that should be specialized are created as virtual methods.

    In practice, this class should never be instantiated.
    """

    def __init__(self, *, expr, tfuncs, labels=[], **kwargs):
        r"""Constructor method.

        Parameters
        ----------
        expr : list of Tuple[Transformation, Interaction]
            list of IT terms to create an IT expression. It **must** be a
            python built-in list.
            
            An IT term is the tuple :math:`(t, p)`, where
            :math:`t : \mathbb{R} \rightarrow \mathbb{R}` is a unary function
            called **transformation** function,  and :math:`p \in \mathbb{R}^d`
            is a vector of size :math:`d`, where :math:`d` is the number of
            variables of the problem. The tuple contains the information to
            create an expression:

            :math:`ITterm(x) = t \circ p(x),`

            and :math:`p` is the **interaction** of the variables:
            
            :math:`p(x) = \prod_{i=1}^{d} x_{i}^{k_{i}}.`

            Each IT term is a tuple containing the name of the transformation
            function and a list of exponents to be used in the interaction
            function.

            The whole expression is evaluated as 

            :math:`f(x) = \sum_{i=1}^{n} w_i \cdot t_i \circ p_i(x),`

            with :math:`w_i` being a coefficient to be adjusted with the
            ``fit()`` method, and :math:`n` the number of terms.

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
        """

        self.expr    = expr
        self.labels  = labels
        self.tfuncs  = tfuncs
        self.n_terms = len(expr)

        # attributes that are changed by fit method
        self._is_fitted = False
        self._fitness   = np.inf 


    def to_str(self, places=3, term_separator=None):
        r"""Method to represent the IT expression as a string.

        The variable names used are the ones given to ``labels`` in the 
        constructor.

        Some simplifications are made to omit trivial operations:

        - if a variable has zero as an exponent, it is omitted (since it will
          eval to 1 regardless of the x value);
        - if the coefficient (or all coefficients, in the multi-class task)
          is zero, the whole term is omitted.
        
        Parameters
        ----------
        places : int, default=3
            Number of decimal places to round the coefficients when printing
            the expression.

        term_separator : string or None, default=None
            string that will be used to contatenate each term. Suggestions
            are ``['\n', ' + ', ', ']``. If set to None, then the separator
            used is ``' + '`` .
        """

        if term_separator == None:
            term_separator = " + "

        # If is not fitted, the method will use placeholder coefs and intercept
        coefs = np.ones(self.n_terms)
        intercept = np.array(0.0)

        if self._is_fitted:
            coefs     = self.coef_
            intercept = self.intercept_
            
        str_terms = []
        for w, (fi_str, ti) in zip(coefs.T, self.expr):
            if np.all(w == 0):
                continue
            
            w_str = f"{w.round(places)}*"

            t_str = " * ".join([
                f"placeholder_{i}" + (f"^{t}" if t!=1 else "")
                for i, t in enumerate(ti) if t!=0
            ])

            str_terms.append(f"{w_str}{fi_str}({t_str})")

        expr_str = term_separator.join(str_terms)

        if len(self.labels)>0:
            for i, l in enumerate(self.labels):
                expr_str = expr_str.replace(f"placeholder_{i}", l)
        else:
            expr_str = expr_str.replace(f"placeholder_", "x_")

        return expr_str  + f"{term_separator}{intercept.round(places)}"


    def _eval(self, X):
        r"""Method to evaluate each IT term on a given data set.

        This makes the mapping from the original variable space to the
        IT expression space.

        Parameters
        ----------
        X : numpy.array of shape (n_samples, n_features)
            data set to be evaluated

        Returns
        -------
        Z : numpy.array of shape (n_samples, n_terms)
            the Z matrix will have one column for each IT term in the
            expression, where the column ``Z[:, i]`` is the evaluation of
            the i-th term for all samples.

            This translates to:

            :math:`Z_{(:, i)} = t_i \circ p_i(x).`
        """

        Z = np.zeros( (len(X), self.n_terms) )

        for i, (fi, ti) in enumerate( self.expr ):
            Z[:, i] = self.tfuncs[fi]( np.prod(np.power(X, ti), axis=1) )

        return Z


    def __str__(self):
        """Overload of the ``__str__`` method. Calls ``itexpr.to_string()``
        method with the default values for the arguments.
        """

        return self.to_str()


    def complexity(self):
        """Method to calculate the IT expression size as if it was an expression
        tree, like the conventional representation for symbolic regression.

        Some simplifications will be made (the same that we do in ``to_str()``),
        so the complexity value corresponds to the string returned by the 
        method.

        Returns
        -------
        complexity : int
            the number of nodes that a symbolic tree would have if the
            IT expression was converted to it.
        """

        coefs = np.ones(self.n_terms)
        if hasattr(self, "coef_"):
            coefs = self.coef_

        tlen = 0
        for coef, (fi, ti) in zip(coefs, self.expr):
            if np.all(coef == 0):
                continue

            # coef, multiplication and transformation function
            tlen += 3

            # exponents != [0, 1] always are a 3 node subtree
            tlen += sum([3 for t in ti if t not in [0, 1]])

            # when exponent is 1, then x^1 = x and we consider only 1 node
            tlen += sum([1 for t in ti if t == 1])

            # exponents equals to 0 are discarded, since x^0 = 1

            # multiplication between each variable in the interaction
            tlen += sum([1 for t in ti if t != 0]) - 1
            
        # sum between terms and the intercept
        return tlen + self.n_terms + 1
        

    def gradient(self, X, tfuncs_dx, logit=False):
        r"""Method to evaluate the gradient of the IT expression for all
        data points in ``X``. The gradients are useful for the
        ``ITExpr_explainer`` class, which calculates feature importances
        and generate plots using the gradient information.

        Parameters
        ----------
        X : numpy.array of shape (n_samples, n_features)
            points to evaluate the gradients.

        tfuncs_dx : dict 
            dictionary like ``tfuncs`` , where the key is the name of the
            function (should have the derivatives of every function in
            tfuncs) and the value is a vectorized function
            representing its derivative.

        logit : boolean, default=False
            boolean variable indicating if the IT expression is being used
            as a linear model or as a linear method of a logistic regression
            predictor. When it is true, then we must consider the derivative
            of the logistic regression.

            let :math:`it(x)` be the IT expression. It is used in a logit model:

            :math:`logit(x) = \frac{1}{1 + e^{-it(x)}}`

            The partial derivative needed to calculate the gradient is:

            :math:`\frac{\partial}{\partial x_i} logit(x)`

            :math:`\Rightarrow \frac{e^{it(x)} it'(x)}{(e^{it(x)} + 1)^2}`

        Returns
        -------
        nabla : numpy.array of shape (n_classes, n_samples, n_features)
            returns a 3-dimensional array. For regression and binary
            classification, ``n_classes=1``. Each line of the matrix inside
            ``nabla[i]`` is the gradient evaluated to the corresponding
            sample in X.

            To ilustrate:

            - Gradient of observation i for regression:
              ``gradients(X, tfuncs_dx)[1, i, :]``
            - Gradient of observation i according to coefficients to classify
              the class j in multi-class classification:
              ``gradients(X, tfuncs_dx, logit=True)[j, i, :]``
        """

        X = check_array(X)
        
        # the gradients can be calculated even before fit
        intercept = [0.0]
        if hasattr(self, "intercept_"):
            intercept = self.intercept_

        coefs = np.ones(self.n_terms)
        if hasattr(self, "coef_"):
            coefs = self.coef_

        # if coefs.ndim==1, then it is a regression ITExpr. Let's make it 2d
        if coefs.ndim == 1:
            coefs = coefs.reshape(-1, 1).T

        # Storing the gradient of each term            
        term_gradients = []

        # i iterates over terms, j over variables
        for j in range(X.shape[1]):

            # evaluating the partial derivative w.r.t each variable
            g_partialx = np.zeros( (len(X), self.n_terms) )
            for i, (fi, ti) in enumerate( self.expr ):
        
                ti_aux = np.copy(ti)
                ti_aux[j] = ti_aux[j] - 1
                
                pi = np.prod(np.power(X, ti), axis=1)

                # avoid multiply zero and nan, would result in nan
                if ti[j] == 0:
                    pi_partialx = np.zeros_like(X[:, j])
                else:
                    pi_partialx = ti[j] * np.prod(np.power(X, ti_aux), axis=1)

                g_partialx[:, i] = tfuncs_dx[fi](pi)*pi_partialx

            term_gradients.append(g_partialx)

        gradients = np.array([np.dot(term_gradients, coef).T for coef in coefs])

        if logit:
            it_eval = np.dot(self._eval(X), coefs.T) + intercept

            for c_idx in range(gradients.shape[0]): # number of classes
                for x_idx in range(gradients.shape[2]): # number of variables
                    gradients[c_idx, :, x_idx] = np.divide(
                        np.exp(it_eval)[:, c_idx] * gradients[c_idx, :, x_idx],
                        np.power(np.exp(it_eval)[:, c_idx] + 1.0, 2)
                    )
            
        return gradients


    def covariance_matrix(self, X, y):
        """virtual method to estimate the covariance matrix.
        Should be overridden by sub-classes.
        """

        raise NotImplementedError()


    def fit(self, X, y):
        """virtual fit method. Should be overridden by sub-classes.

        It takes a dictionary as named arguments to allow sub-classes to have
        specific parameters to the fit method.
        """

        raise NotImplementedError()


    def predict(self, X):
        """virtual predict method. Should be overridden by sub-classes.
        """

        raise NotImplementedError()


    def predict_proba(self, X):
        """virtual predict_proba method. Should be overridden by sub-classes.
        """

        raise NotImplementedError()