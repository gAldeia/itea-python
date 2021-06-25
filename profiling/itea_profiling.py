import numpy as np

from itea.regression     import ITEA_regressor
from itea.classification import ITEA_classifier

from sklearn.datasets import make_regression, make_blobs


tfuncs    = {'id' : lambda x: x}
tfuncs_dx = {'id' : lambda x: np.ones_like(x),}


if __name__ == '__main__':

    # Regression execution
    X_reg, y_reg, coef = make_regression(
        n_samples     = 100,
        n_features    = 5,
        n_informative = 5,
        random_state  = 0,
        noise         = 0.0,
        bias          = 100.0,
        coef          = True
    )

    reg = ITEA_regressor(
        gens=100, 
        popsize=100, 
        verbose=10,
        tfuncs=tfuncs,
        tfuncs_dx=tfuncs_dx,
        random_state=42
    ).fit(X_reg, y_reg)

    print(reg.bestsol_)
    print(reg.bestsol_.coef_)

    # Classification execution
    X_clf, y_clf = make_blobs(
        n_samples    = 100,
        n_features   = 2,
        cluster_std  = 1,
        centers      = [(-10,-10), (0,0), (10, 10)],
        random_state = 0,
    )

    clf = ITEA_classifier(
        gens=100, 
        popsize=100, 
        verbose=10,
        tfuncs=tfuncs,
        tfuncs_dx=tfuncs_dx,
        random_state=42
    ).fit(X_clf, y_clf)

    print(clf.bestsol_)
    print(clf.bestsol_.coef_)