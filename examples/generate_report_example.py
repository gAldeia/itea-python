# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 03-09-2021 by Guilherme Aldeia

"""Auxiliary script to generate te refered report in the
agnostic_explainers notebook. The code here is the same in the notebook,
but in a script to be called while building the documentation.
"""

import numpy  as np

from sklearn import datasets

from sklearn.model_selection import train_test_split

from itea.regression import ITEA_regressor
from itea.inspection import ITEA_summarizer

import warnings
warnings.filterwarnings(action='ignore', module=r'itea')

if __name__ == '__main__':
    housing_data = datasets.fetch_california_housing()
    X, y   = housing_data['data'], housing_data['target']
    labels = housing_data['feature_names']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    tfuncs = {
        'log'      : np.log,
        'sqrt.abs' : lambda x: np.sqrt(np.abs(x)),
        'id'       : lambda x: x,
        'sin'      : np.sin,
        'cos'      : np.cos,
        'exp'      : np.exp
    }

    tfuncs_dx = {
        'log'      : lambda x: 1/x,
        'sqrt.abs' : lambda x: x/( 2*(np.abs(x)**(3/2)) ),
        'id'       : lambda x: np.ones_like(x),
        'sin'      : np.cos,
        'cos'      : lambda x: -np.sin(x),
        'exp'      : np.exp,
    }

    reg = ITEA_regressor(
        gens         = 75,
        popsize      = 75,
        max_terms    = 5,
        expolim      = (-1, 1),
        verbose      = 10,
        tfuncs       = tfuncs,
        tfuncs_dx    = tfuncs_dx,
        labels       = labels,
        random_state = 42,
        simplify_method = None
    ).fit(X_train, y_train)

    summarizer = (
        ITEA_summarizer(itea=reg)
        .fit(X_train, y_train)
        .autoreport(save_path='./examples/')
    )