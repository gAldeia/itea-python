# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 05-29-2021 by Guilherme Aldeia


"""
Interaction-Transformation Evolutionary Algorithm implementation
(ITEA) in python for Classification and Regression. The algorithm is based on
scikit-learn guidelines for creating ML classes and can be integrated with
their tools."""


import itea.classification as classification
import itea.regression     as regression
import itea.inspection     as inspection


__all__ = [
    'classification',
    'regression',
    'inspection'
]