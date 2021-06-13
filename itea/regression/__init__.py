# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 05-29-2021 by Guilherme Aldeia


r"""Interaction Transformation Evolutionary Algorithm for **regression**

This sub-module implements a specialization of the base classes ``BaseITEA``
and ``BaseITExpr`` to be used on regression tasks.

Ideally, the user should import and use only the ``ITEA_regressor``
implementation, while the ``ITExpr_regressor`` should be created by means of the
itea instead of manually by the user.

The ``ITExpr_regressor`` works just like any fitted scikit regressor,
but --- in order to avoid the creation of problematic expressions --- I
strongly discourage the direct instantiation of ``ITExpr_regressor``.
"""


from itea.regression._ITExpr_regressor import ITExpr_regressor
from itea.regression._ITEA_regressor   import ITEA_regressor


__all__ = [
    'ITEA_regressor'
]