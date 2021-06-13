# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 05-29-2021 by Guilherme Aldeia


r"""Interaction Transformation Evolutionary Algorithm for **classification**

This sub-module implements a specialization of the base classes ``BaseITEA``
and ``BaseITExpr`` to be used on classification tasks.

Ideally, the user should import and use only the ``ITEA_classifier``
implementation, while the ``ITExpr_classifier`` should be created by means of
the itea instead of manually by the user.

The ``ITExpr_classifier`` works just like any fitted scikit classifier,
but --- in order to avoid the creation of problematic expressions --- I
strongly discourage the direct instantiation  of ``ITExpr_classifier``.
"""


from itea.classification._ITExpr_classifier import ITExpr_classifier
from itea.classification._ITEA_classifier   import ITEA_classifier


__all__ = [
    'ITEA_classifier'
]