# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 05-30-2021 by Guilherme Aldeia


"""Generator of random IT expressions.

The generator guarantee that no trivial collection of IT terms will be created,
but this doesn't necessarily means that all expressions will be good. The
expressions can predict nan values after this stage, and a better quality
can be ensured through the sanitizer and simplifier methods.
"""


import numpy as np


__all__ = [
    'uniform'
]


def uniform(max_terms, expolim, tfuncs, nvars, random_state, **kwargs):
    """Creating expressions by uniformly choosing the configurations.

    returns a list of Tuple[Transformation, Interaction].
    """

    while True:
        nterms = random_state.randint(1, max_terms + 1)

        funcs  = random_state.choice(list(tfuncs.keys()), size=nterms)
        terms  = random_state.randint(
            expolim[0], expolim[1] + 1, size=(nterms, nvars))

        # To avoid trivial expressions at a low computational cost, I will
        # just set one exponent to one.
        for i in range(nterms):
            if np.all(terms[i]==0):
                terms[i, random_state.randint(nvars)] = 1 

        yield list(zip(funcs.tolist(), terms.tolist()))