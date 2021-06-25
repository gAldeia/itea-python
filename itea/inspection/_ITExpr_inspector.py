# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.1
# Last modified: 06-17-2021 by Guilherme Aldeia


"""ITExpr_inspector class.
"""


import numpy     as np

from scipy import stats

from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics          import mutual_info_score


class ITExpr_inspector():
    """class ITExpr_inspector.
    
    Based on a more statistical approach, this class 
    implements methods to measure the quality of the final expression by
    calculating information between individual terms.
    """

    def __init__(self, *, itexpr, tfuncs, decimal_places=3):
        """Constructor method.
        
        Parameters
        ----------
        itexpr : ITExpr_regressor or ITExpr_classifier
            fitted instance of an ``ITExpr`` class to be explained.
        
        tfuncs : dict
            transformations functions. Should always
            be a dict where the keys are the names of the transformation
            functions and the values are unary vectorized functions.
        """

        self.itexpr         = itexpr
        self.tfuncs         = tfuncs
        self.decimal_places = decimal_places


    def fit(self, X, y):
        """Fit method to store the data used in the training of the given
        itexpr instance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            data used to train the itexpr model.

        y : array-like of shape (n_samples, )
            target data used to train the itexpr model.

        Returns
        -------
        self : ITExpr_inspector
            inspector with the calculated covariance matrix.
        """

        X = check_array(X)
        
        self.X_ = X
        self.y_ = y


        self.varcovar_beta = self.itexpr.covariance_matrix(X, y)
        
        if self.varcovar_beta.ndim == 2:
            self.varcovar_beta = np.array([self.varcovar_beta])

        # Matrix containing each term evaluation on training data
        self.Z = np.ones( (X.shape[0], self.itexpr.n_terms + 1) )
        self.Z[:, :-1] = self.itexpr._eval(X)

        return self   


    def _coef_stderr(self):
        """Method for estimating the standard error of the coefficients.

        The estimated standard deviations will be calculated by taking the
        square root from the main diagonal.
        """
        
        stderrs = []
        for i in range(self.varcovar_beta.shape[0]):
            stderrs.append(
                np.sqrt(np.diag(self.varcovar_beta[i]))
            )

        if len(stderrs) == 1:
            stderrs = stderrs[0]
        
        return [str(stderr.round(self.decimal_places))
                for stderr in np.array(stderrs).T]


    def _disentanglement(self):
        """Method for calculating the mean disentanglement for each term.
        
        The mean disentanglement is the mean Pearson's correlation between
        the term of interest and the remaining terms.

        The disentanglement (measured by the collinearity between the generated
        features) was proposed in "Learning feature spaces for regression with
        genetic programming". The idea is that, when creating new features, a
        disentangled representation ideally contains a minimal set of features.
        In this paper, the authors tries to minimize collinearity between
        features in order to promote disentanglement.

        This metric is reported to indicate if there is a high degree of
        disentanglement on the expression.

        Notes
        -----
        This calculation was proposed in
        "La Cava, W., Moore, J.H. Learning feature spaces for regression with
        genetic programming. Genet Program Evolvable Mach 21, 433–467 (2020)"
        """
        
        if self.itexpr.n_terms == 1:
            return [0.0]

        disentanglements = []
        for col in range(self.itexpr.n_terms):

            col_disentanglement = []
            for col_to_compare in range(self.itexpr.n_terms):

                if col != col_to_compare:
                    corr, p = stats.pearsonr(
                        self.Z[:, col], self.Z[:, col_to_compare])
                    
                    # Pearson's correlation divides by the std, and the
                    # existante of a result is not guaranted. We'll consider
                    # a zero correlation in this cases
                    corr = 0.0 if np.isnan(corr) else corr**2
                    
                    col_disentanglement.append(corr)

            disentanglements.append(
                np.mean(col_disentanglement).round(self.decimal_places))

        return disentanglements


    def _pred_var(self):
        """Method to calculate the variance of the predictions each term
        produces on the training data.
        """

        if hasattr(self.itexpr, 'classes_'):
            variances = []

            for coef, intercept in zip(
                self.itexpr.coef_, self.itexpr.intercept_):
                
                coef_and_intercept = np.append(
                    coef, intercept)
                    
                variances.append(np.var(
                    self.Z * coef_and_intercept,
                axis=0).round(self.decimal_places))

            return [str(v.round(self.decimal_places))
                    for v in np.array(variances).T]
        else:
            coef_and_intercept = np.append(
                self.itexpr.coef_, self.itexpr.intercept_)

            return np.var(
                self.Z * coef_and_intercept, axis=0).round(self.decimal_places)


    def _continuous_mutual_info(self):
        """Method to calculate the mean continuous mutual information for 
        each term. The mutual information is calculated between the term of
        interest and the remaining terms.
        """

        if self.itexpr.n_terms == 1:
            return [0.0]
            
        mutual_informations = []
        for col in range(self.itexpr.n_terms):

            col_mutual_information = []
            for col_to_compare in range(self.itexpr.n_terms):

                if col != col_to_compare:
                    bins = int(np.floor(np.sqrt(len(self.X_))))
                    
                    # first element is a histogram
                    c_xy = np.histogram2d(
                        self.Z[:, col],
                        self.Z[:, col_to_compare],
                        bins
                    )[0]
                    
                    col_mutual_information.append(
                        mutual_info_score(None, None, contingency=c_xy)
                    )

            mutual_informations.append(
                np.mean(col_mutual_information).round(self.decimal_places))

        return mutual_informations


    def terms_analysis(self):
        """Method to calculate different metrics for the terms composing
        the IT expression.
        
        Returns
        -------
        analysis : dict
            returns a dictionary containing several term information
            and metrics calculated for each term:

            - coef: coefficient of each term (or coefficients, if the
              itexpr is an instance of ``ITExpr_classifier``);
            - func: transformation function of each term;
            - strengths: the exponents of each term;
            - coef stderr.: the standard error of the coefficients;
            - mean pairwise disentanglement: the mean disentanglement between
              each term when compared with the others;
            - mean mutual information: the mean continuous mutual information
              between each term when compared with the others;
            - prediction var.: the variance of the predicted outcomes for
              each term when predicting the training data.
        """

        check_is_fitted(self)
        
        coefs, funcs, strengths = [], [], []
        for wi, (fi, ti) in zip(self.itexpr.coef_.T, self.itexpr.expr):
            
            funcs.append(fi)
            strengths.append(str(ti))
            
            coefs.append(str(wi.round(self.decimal_places)))

        # Calculating statistics related to the intercept
        coefs        = coefs + [
            str(np.round(self.itexpr.intercept_, self.decimal_places))]
        funcs        = funcs + ['intercept']
        strengths    = strengths + ['---']
        disentangles = self._disentanglement() + [0.0]
        mutual_info  = self._continuous_mutual_info() + [0.0]
        
        return {
            'coef'                           : coefs,
            'func'                           : funcs,
            'strengths'                      : strengths, 
            'coef\nstderr.'                  : self._coef_stderr(),
            'mean pairwise\ndisentanglement' : disentangles,
            'mean mutual\ninformation'       : mutual_info,
            'prediction\nvar.'               : self._pred_var(),
        }