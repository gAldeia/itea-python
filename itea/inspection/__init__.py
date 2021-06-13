# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.1
# Last modified: 06-07-2021 by Guilherme Aldeia


"""Interaction Transformation expression's **Inspector**

Sub-module containing three classes to help inspect and explain the
results obtained with the itea.

- ``ITExpr_explainer``: Implementations of feature importances methods specific
  to the Interaction-Transformation representation, and several visualization
  tools to help interpret the final expression;
- ``ITExpr_inspector``: Based on a more statistical approach, this class 
  implements methods to measure the quality of the final expression by
  calculating information between individual terms;
- ``ITExpr_texifier``: Creation of latex representations of the final expression
  and its derivatives. In cases where the final expression is simple enough,
  the analysis of the expression can provide useful insights.

All the modules are designed to work with `ITExpr`s. After the evolutionary
process is performed (by calling `fit()` on the `ITEA_classifier` or
`ITEA_regressor`), the best final expression can be accessed by
`itea.bestsol_`, and those classes are specialized in different ways of
inspecting the final model.

Additionally, there is one class designed to work with the ´`itea``, instead
of ``ITExpr`` expressions. The class ``ITEA_summarizer`` implements a method
to automatically create a pdf file containing information generated with 
all the inspection classes, in an attempt to automate the task of generating
an interpretability report. 
"""


from itea.inspection._ITExpr_explainer import ITExpr_explainer
from itea.inspection._ITExpr_inspector import ITExpr_inspector
from itea.inspection._ITExpr_texifier  import ITExpr_texifier
from itea.inspection._ITEA_summarizer  import ITEA_summarizer

import jax


# Must be used at startup. We'll perform lightweight usage with jax 
jax.config.update('jax_platform_name', 'cpu')


__all__ = [
    'ITExpr_explainer',
    'ITExpr_inspector',
    'ITExpr_texifier',
    'ITEA_summarizer'
]