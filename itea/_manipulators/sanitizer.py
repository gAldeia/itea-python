# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 05-30-2021 by Guilherme Aldeia


"""Sanitizer method to remove repeated terms that can be randomly created.
"""


def sanitize(expr):
    """It takes an expr (the list of IT terms, not an ITExpr class) and 
    cleans it by removing repeated terms.

    This is done because there are two points where repeated terms can be
    created: during the generation or after the mutation.

    Since the generator guarantees that every expr will have at least one 
    IT term, the result returned by sanitize will always be a valid and non-
    trivial expression.    
    """

    n_terms     = len(expr)
    unique_expr = []

    for i, (fi, ti) in enumerate(expr):
        include=True

        for _, (fj, tj) in enumerate(expr[i+1:]):
            if fi==fj and ti == tj:
                include = False
                break

        if include:
            unique_expr.append(expr[i])

    return unique_expr