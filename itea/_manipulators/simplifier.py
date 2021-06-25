# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.1
# Last modified: 06-17-2021 by Guilherme Aldeia


"""Methods to simplify expressions. Methods should take an instance of
``ITExpr``, performs the modifications inplace and return the same instance
of ``ITExpr``.

This method works inplace to make its usage simpler by dispensation of
lots of arguments.

It is important to notice that, when a change is made, the instance should
be modified to indicate the necessity of a new fit. This is achieved by 
changing the private ``itexpr._is_fitted`` attribute.

All simplification methods should use named arguments.
"""


import numpy as np


__all__ = [
    'simplify_by_coef',
    'simplify_by_var',
]


def _keep_selected(itexpr, selected_terms):
    """This method takes an ITExpr instance and a list containing the indexes
    of the selected terms. All simplification methods should do checkings and
    find out what they will preserve and discard.
    """

    itexpr.expr = [itexpr.expr[i] for i in selected_terms]

    itexpr.n_terms    = len(selected_terms)
    itexpr._is_fitted = False

    return itexpr


def simplify_by_coef(*, itexpr, **kwargs):
    """Simplification of an ITExpr based on the coefficients of each term.
    
    The simplification threshold is set to a default value and every term
    that has a smaller absolute coefficient (or an array of coefficients, in
    case of multi-class classification) are discarded.
    """

    coefs = itexpr.coef_

    # Transposing in multi-class case to correctly iterate over the coefs
    if coefs.ndim == 2:
        coefs = coefs.T

    selected_terms = []
    for i, coef in enumerate(coefs):
        if np.any(np.abs(coef) >= 1e-5):
            selected_terms.append(i)

    # We'll keep the term if it is the only on the ITExpr
    if len(selected_terms) < 1:
        return itexpr

    return _keep_selected(itexpr, selected_terms)


def simplify_by_var(*, itexpr, X, **kwargs):
    """Simplification by the variance of the predictions that each term
    has based on the training samples. A term that presents a very small
    prediction variance will behave almost like an intercept, and can occur
    in some data sets.

    This method evaluates the variance of each term and them normalizes the
    variance to represent the relative variance when compared to the other
    terms.
    """

    variances = [np.var(itexpr.tfuncs[fi]( np.prod(np.power(X, ti), axis=1) ))
                 for fi, ti in itexpr.expr]

    tot_variance = np.sum(variances)
    selected_terms = []
    
    for i, v in enumerate(variances):
        if v/tot_variance >= 1e-2:
            selected_terms.append(i)

    if len(selected_terms) < 1:
        return itexpr

    return _keep_selected(itexpr, selected_terms)