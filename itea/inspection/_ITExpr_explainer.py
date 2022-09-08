# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.9
# Last modified: 11-24-2021 by Guilherme Aldeia


"""Model-specific interpretability methods.
"""


import warnings

import numpy             as np
import matplotlib.pyplot as plt

from jax                      import grad, vmap
from sklearn.utils.validation import check_array, check_is_fitted
from matplotlib.gridspec      import GridSpecFromSubplotSpec 

from sklearn.exceptions import NotFittedError

from scipy.stats.mstats import mquantiles
from matplotlib         import transforms

class ITExpr_explainer():
    """Class to explain ITExpr expressions.
    """

    def __init__(self, *, itexpr, tfuncs, tfuncs_dx=None):
        """Constructor method.

        Parameters
        ----------
        itexpr : ITExpr_regressor or ITExpr_classifier
            fitted instance of an ``ITExpr`` class to be explained.
        
        tfuncs : dict
            transformations functions. Should always
            be a dict where the keys are the names of the transformation
            functions and the values are unary vectorized functions.

        tfuncs_dx : dict, default=None
            derivatives of the
            given transformations functions, the same scheme.
            When set to None, the itea package will use automatic
            differentiation  through jax to create the derivatives.

        """
        
        self.itexpr    = itexpr
        self.tfuncs    = tfuncs
        self.tfuncs_dx = tfuncs_dx


    def _check_args(self):
        """Method to check consistency between tfuncs and tfuncs_dx,
        and verify if the itexpr passed to the constructor was alreay fitted.

        Raises
        ------
            NotFittedError
                If the given ``itexpr`` is not fitted.

            KeyError
                If not all keys of ``tfuncs`` are contained
                in the keys of ``tfuncs_dx``.
        """
        
        if not self.itexpr._is_fitted:
            raise NotFittedError("The itexpr was not fitted.")

        # Creaing the partial derivatives if none was given.
        if not isinstance(self.tfuncs_dx, dict):
            warnings.warn("It wasn't specified a dict for tfuncs_dx. "
                          "They will be automatically generated using Jax. "
                          "For this, make sure that the tfuncs uses the "
                          "jax.numpy instead of numpy to create the "
                          "transformation functions. You can access the "
                          "automatic derivatives with explainer.tfuncs_dx.")

            self.tfuncs_dx = dict()

            for k, v in self.tfuncs.items():
                self.tfuncs_dx[k] = vmap(grad(v))

        if not set(self.tfuncs.keys()) <= set(self.tfuncs_dx.keys()):
            raise KeyError("Not all functions in tfuncs have their "
                           "corresponding derivative in tfuncs_dx. "
                           "You need to create all derivatives, or pass "
                           "tfuncs_dx=None as argument in the constructor.")


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
        self : ITExpr_explainer
            explainer with a calculated covariance matrix, ready to generate
            plots and explanations.

        Raises
        ------
            NotFittedError
                If the given ``itexpr`` is not fitted.

            KeyError
                If not all keys of ``tfuncs`` are contained
                in the keys of ``tfuncs_dx``.

        """

        X = check_array(X)

        self._check_args()
        
        self.X_ = X
        self.y_ = y

        self.varcovar_beta = self.itexpr.covariance_matrix(X, y)
        
        # To have a more generic manipulation, we'll cast regression and
        # binary classification matrices to a 3d shape
        if self.varcovar_beta.ndim == 2:
            self.varcovar_beta = np.array([self.varcovar_beta])

        return self

    
    def selected_features(self, idx=False):
        """Method to identify if any of the original features were left out
        of the IT expression during the evolution of the exponents' array.

        Parameters
        ----------
        idx : bool, default=False
            boolean variable specifying if the method should return a list
            with the labels of the features or their indexes.

        Returns
        -------
        selected : array of shape (n_selected_features)
            array containing the labels of the features that are present in
            the model, or their indexes if ``idx=True``.
        """

        terms = [ti for (fi, ti) in self.itexpr.expr]

        selected = np.where(np.sum(np.abs(terms), axis=0) != 0)[0]

        if len(self.itexpr.labels) > 0 and not idx:
            return np.array(self.itexpr.labels)[selected]

        return selected


    def average_partial_effects(self, X):
        r"""Feature importance estimation through the Partial Effects method.

        This method attributes the importance to the i-th variable by
        using the mean value of the partial derivative w.r.t. i, evaluated for
        all data in X.

        This method assumes that every feature is continuous.

        The partial effect of a given variable for a function :math:`f` is
        calculated by:

        :math:`PE_j = \frac{\partial \widehat{f}}{\partial x_j}
        (\mathbf{x}, \mathbf{\beta}).`

        The IT expression can be automatically differentiated:

        :math:`\frac{\partial  \widehat{f}(x)}{\partial x_j} = w_1
        \cdot it'_1(x) + \ldots + w_m \cdot it'_m(x),`

        where

        :math:`it'_i(x) = g'_i(p_i(x)) \cdot p'_i(x),`

    	:math:`p'_i(x) = k_j\frac{p_i(x)}{x_j}.`

        Parameters
        ----------
        X : numpy.array of shape (n_samples, n_features)
            data from which we want to extract the feature importance for
            the predicted outcomes of the model. If X is a single observation,
            the partial effects are calculated and returned directly. If X 
            contains multiple observations, then the mean partial effect for
            each variable through all observations is returned.

        Returns
        -------
        ape_values : numpy.array of shape (n_classes, n_features)
            the importance of each feature of X, for each class found in the
            given itexpr. If itexpr is an ``ITExpr_regressor``, then the
            returned array will have the shape (1, n_features).

        Notes
        -----
        This feature importance measure is based on the paper:
        "Aldeia, G. & França, F. (2021).
        Measuring Feature Importance of Symbolic Regression Models
        GECCO."
        """

        check_is_fitted(self)
        X = check_array(X)

        # reminder: gradients[class, # obs, variable]
        gradients = self.itexpr.gradient(
            X, self.tfuncs_dx, logit=hasattr(self.itexpr, 'classes_'))

        return np.mean(np.abs(gradients), axis=1)


    def shapley_values(self, X):
        r"""Feature importance estimation through approximation of the 
        Shapley values with the gradient information.

        The shapley values come from the coalitional game theory and were
        proposed as a feature importance measures by Scott Lundberg in 
        "Scott M. Lundberg and Su-In Lee. 2017. A unified approach to
        interpreting model predictions. NIPS". The equation:

        :math:`\phi_j(x) = \sum_{Q \subseteq S \setminus \left\{j\right\}}
        {\frac{|Q|!(|S| - |Q| - 1)!}{|S|!}
        (\Delta_{Q \cup \{j\}}(x) - \Delta_{Q}(x))},`

        where :math:`S` is the set of variables, :math:`\Delta_Q` is the
        contribution of the set :math:`Q`, the difference between the
        prediction when we fix the values of :math:`Q`, and the expected
        prediction.
        
        It is possible to approximate these values by the equation:

        :math:`\widehat{\phi_j}(x) = \mathbb{E}[PE_j]
        \cdot (x_j - \mathbb{E}[x_j]),`

        Where :math:`PE_j` is the partial effect w.r.t :math:`j`.

        Parameters
        ----------
        X : numpy.array of shape (n_samples, n_features)
            data from which we want to extract the feature importance for
            the predicted outcomes of the model. If X is a single observation,
            the shapley values are calculated and returned directly. If X 
            contains multiple observations, then the mean shapley value for
            each variable through all observations is returned.

        Returns
        -------
        shapley_values : numpy.array of shape (n_classes, n_features)
            the importance of each feature of X, for each class found in the
            given itexpr. If itexpr is an ``ITExpr_regressor``, then the
            returned array will have the shape (1, n_features).

        Notes
        -----
        The shapley values are described and explained in:
        "Scott M. Lundberg and Su-In Lee. 2017. A unified approach to
        interpreting model predictions. NIPS"

        The approximation of shapley values by means of the partial effect
        was studied in the paper:
        This feature importance measure is based on the paper:
        "Aldeia, G. & França, F. (2021).
        Measuring Feature Importance of Symbolic Regression Models
        GECCO."
        """

        check_is_fitted(self)
        X = check_array(X)

        # gradients are evaluated with training data
        # reminder: gradients[class, # obs, variable]
        gradients = self.itexpr.gradient(
            self.X_, self.tfuncs_dx, logit=hasattr(self.itexpr, 'classes_'))

        importances = np.zeros( (gradients.shape[0], X.shape[0], X.shape[1]) )

        # iterate through each variable for each class
        for c in range(len(gradients)):
            for i in range(X.shape[1]):
                importances[c, :, i] = \
                    gradients[c, :, i].mean() * (X[:, i] - self.X_[:, i].mean())

        return np.mean(np.abs(importances), axis=1)


    def integrated_gradients(self, X, m_steps=300):
        r"""Feature importance estimation through the Integrated Gradients
        method.

        The idea of the integrated gradient is to calculate a local
        importance score for a feature :math:`i` by evaluating ---
        given a baseline :math:`\mathbf{x}'` and a specific point
        :math:`\mathbf{x}` --- the integral of the models' gradients
        :math:`\frac{\partial f}{\partial x_i}` along a straight
        line between the baseline and the specific point. Since gradients
        describe how minimal local changes interfere with the
        model's predictions, the calculated importance represents the
        accumulation of the gradient of each variable to go from the baseline
        to the specific point.

        :math:`IntegratedGrads_{i}^{approx} (\mathbf{x}) =
        \frac{(x_i - x_{i}')}{m} \sum_{k = 1}^{m}
        \frac{\partial \widehat {f} (\mathbf{x}' +
        \frac{k}{m}(\mathbf{x} - \mathbf{x}'))} {\partial x_i}`

        In our implementation, the baseline used is the average value of the
        training attributes used in ``fit(X_train, y_train)``, thus the
        calculated value is close to SHAP.

        Parameters
        ----------
        X : numpy.array of shape (n_samples, n_features)
            data from which we want to extract the feature importance for
            the predicted outcomes of the model. If X is a single observation,
            the integrated gradients are calculated and returned directly. If X 
            contains multiple observations, then the mean integrated gradients
            for each variable through all observations is returned.

        m_steps : int, default=300
            number of steps :math:`m` used in the Riemman summation
            approximation of the integral. Small values leads to a poor
            approximation, while higher values decreases the error at the cost
            of a higher computational cost.

        Returns
        -------
        ig : numpy.array of shape (n_classes, n_features)
            the importance of each feature of X, for each class found in the
            given itexpr. If itexpr is an ``ITExpr_regressor``, then the
            returned array will have the shape (1, n_features).

        Notes
        -----
        The Integrated Gradient method was proposed in the paper:
        "Mukund Sundararajan, Ankur Taly, and Qiqi Yan. 2017. 
        Axiomatic attribution for deep networks. In Proceedings of the
        34th International Conference on Machine Learning - Volume
        70 (ICML'17). JMLR.org, 3319–3328."
        """

        check_is_fitted(self)
        X = check_array(X)

        x_baseline = np.mean(self.X_, axis=0).reshape(1, -1)
        
        # Generate m_steps intervals for integral approximation below.
        alphas = np.linspace(start=0.0, stop=1.0, num=m_steps+1).reshape(-1, 1)
        
        # This will be the final IG explanation
        if hasattr(self.itexpr, 'classes_'):
            n_classes = len(self.itexpr.classes_)
        else:
            n_classes = 1
            
        integrated_gradients = np.zeros( (n_classes, X.shape[0], X.shape[1]) )
        
        # Lets calculate the IG for each observation and take the mean
        # to explain. This is done to make it possible to estimate local and
        # global explanations, where the global is the mean over all given
        # observations.
        for obs_idx in range(X.shape[0]):
            x = X[obs_idx, :].reshape(1, -1)
            
            delta = x - x_baseline

            interpolated = x_baseline + (delta * alphas)

            # equation gradients, used in IG
            # reminder: gradients[class, # obs, variable]
            gradients = self.itexpr.gradient(interpolated, self.tfuncs_dx)

            # for each class 
            for c in range(len(gradients)):

                # Integral approximation using Riemman summation. This is the
                # trapezoidal sum
                riemann_sum = (gradients[c, :-1, :] + gradients[c, 1:, :])/2.0

                # Taking the mean and scaling
                integrated_gradients[c, obs_idx] = (
                    riemann_sum.mean(axis=0) * (x - x_baseline))

        return np.abs(np.mean(integrated_gradients, axis=1))
    


    def plot_feature_importances(self,
        X,
        *,
        barh_kw            = None,
        ax                 = None,
        grouping_threshold = 0.05,
        target             = None,
        importance_method  = 'pe',
        show_avg_values    = True,
        show               = True,
    ):
        """Bar plot of the feature importance, that can be calculated with
        the Partial effects (PE) or Shapley Values (shapley).

        .. image:: assets/images/plot_feature_importance_1.png
            :align: center
        .. image:: assets/images/plot_feature_importance_2.png
            :align: center

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            data to explain.

        bar_kw : dict or None, default=None
            dictionary with keywords to be used when generating the plots.
            When set to None, then ``bar_kw= {'alpha':1.0, 'align':'center'}``.

        ax : matplotlib.axes or None, default=None
            axis to generate the plot. If none is given, then a new axis is
            created. if is a single axis, the plot will be drawn within the
            given axis. The new axis and figure created can be accessed through
            the ``axes_`` and ``figure_`` attributes. Notice that, if multiple
            plot functions were called, then ``axes_`` and ``figure_``
            corresponds to the latest plot generated.

        grouping_threshold : float, default = 0.05
            The features will be iterated in order of importance, from the
            smallest to the highest importance, and a group of the smallest
            features that sum up less than the given percentage of importance
            will be grouped to reduce the plot information.
            To disable the creation of the group, set ``grouping_threshold=0``.

        target : string, int, list[strings], list[ints] or None, default=None
            The targets to be considered when generating the plot for 
            ``ITExpr_classifier``. If the training data used strings as targets
            on ``y``, then the target must be a string of a valid class or a
            list of valid classes. If the training data was encoded as integers,
            then the target must be an int or a list of integers.
            This argument is ignored if the itexpr is an ``ITExpr_regressor``.

        importance_method : string, default='pe'
            string specifying which method should be used to estimate feature
            importance. Available methods are: ``['pe', 'shapley', 'ig']``.

        show_avg_values : bool, default=True
            determines if the average feature value should be displayed 
            left to the feature names in the y axis.
        
        show : bool, default=True
            boolean value indicating if the generated plot should be displayed
            or not.

        Raises
        ------
            ValueError
                If ``ax`` or ``target`` has invalid values.

        Notes
        -----
        This plot is heavily inspired by the `bar plot from the SHAP package
        <https://shap.readthedocs.io/en/latest/
        example_notebooks/api_examples/plots/bar.html>`_
        
        """

        check_is_fitted(self)
        X = check_array(X)

        # handling the ax attribute
        if ax is None:
            self.figure_, self.axes_ = plt.subplots()
        elif not isinstance(ax, plt.Axes):
                raise ValueError(
                    f'Expected ax to have 1 axes, got {np.asarray(ax).size}')
        else:
            self.figure_ = ax.figure
            self.axes_   = ax

        # picking the importance method
        importance_methods = {
            'shapley': self.shapley_values,
            'pe'     : self.average_partial_effects,
            'ig'     : self.integrated_gradients
        }

        if importance_method in importance_methods.keys(): 
            importance_f = importance_methods[importance_method]
        else:
            raise ValueError(
                f"Expected importance_method to be 'pe', 'shapley' or 'ig'. "
                f"got {importance_method}"
            )

        # finding out if itexpr is a classification or regression specialization
        if hasattr(self.itexpr, 'classes_') and np.size(self.itexpr.classes_)>2:
            if target is None:
                target= self.itexpr.classes_
            
            target=np.array([target]).flatten()

            if not set(target).issubset(set(self.itexpr.classes_)):
                raise ValueError(f'target not in est.classes_, got {target}')
   
            target_idx = [list(self.itexpr.classes_).index(t) for t in target]

            importances = importance_f(X)[target_idx]
        else:
            importances = importance_f(X).reshape(1, -1)
                
        # classifying the features importances
        mean_values = np.abs(np.mean(X, axis=0))
        if len(self.itexpr.labels) > 0:
            y_ticks_labels = np.array([
                f'{round(m, 3)} = {l}' if show_avg_values else f'{l}'
                for m, l in zip(mean_values, self.itexpr.labels)])
        else:
            y_ticks_labels = np.array([
                f'{round(m, 3)} = x_{i}' if show_avg_values else f'x_{i}'
                for i, m in enumerate(mean_values)])
        
        order = np.argsort(-np.sum(importances, axis=0))

        # grouping the least important features that don't exceed the threshold
        others = np.zeros( (len(importances), 1) )
        features_in_others = 0
        tot_importances = np.sum(importances)
        for i in order[::-1]:
            if (np.sum(others + np.sum(importances, axis=0)[i])/tot_importances
            ) < grouping_threshold:

                others[:, 0] += importances[:, i]
                features_in_others += 1

        # Greater than one because a group of one is not a group
        if features_in_others > 1:
            final_importances = np.hstack((
                importances[:, order[:len(order)-features_in_others]],
                others
            ))

            labels = np.hstack((
                y_ticks_labels[order[:len(order)-features_in_others]],
                [f'Other features ({features_in_others})']
            ))
        else:
            final_importances = importances[:, order]
            labels = y_ticks_labels[order]

        default_kw = {'alpha':1.0, 'align':'center'}

        if barh_kw is None:
            barh_kw = default_kw
        else:
            barh_kw = {**default_kw, **barh_kw}

        self.axes_.grid(axis='y', zorder=-1, ls=':')
        self.axes_.set_axisbelow(True)
        
        # plotting stacked bars when target is a list
        checkpoint = np.zeros_like(final_importances.shape[1])
        for i, final_importance in enumerate(final_importances):
            self.axes_.barh(
                np.arange(final_importance.shape[0]),
                final_importance[::-1], left=checkpoint,
                label=(
                    self.itexpr.classes_[target_idx[i]]
                    if hasattr(self.itexpr, 'classes_') else None),
                **barh_kw)
            checkpoint = np.add(checkpoint, final_importance[::-1])

        self.axes_.set_yticks(range(len(labels)))
        self.axes_.set_yticklabels(labels[::-1])

        # Setting the xlabel to inform if it is a single observation or multiple 
        if X.shape[0] > 1:
            self.axes_.set_xlabel(f'average(|{importance_method} values|)')
        else:
            self.axes_.set_xlabel(f'|{importance_method} values|')

        # annotating the importance values
        offset = np.max(np.sum(final_importances, axis=0))/50
        for i, rect in enumerate(
            self.axes_.patches[-(final_importances.shape[1]):]):
            
            self.axes_.text(
                rect.get_width()+rect.get_x()+offset,
                rect.get_y() + 0.5,
                np.sum(final_importances, axis=0)[-(i+1)].round(2),
                ha='left',
                va='top'
            )

        self.axes_.spines['right'].set_visible(False)
        self.axes_.spines['top'].set_visible(False)
        self.axes_.set_xlim(
            (None, np.max(np.sum(final_importances, axis=0))+5*offset) )

        if hasattr(self.itexpr, 'classes_'):
            self.axes_.legend(loc=4)

        if show:
            plt.show()
        

    def _evaluate_partial_effects_at_means(self, X, percentiles, num_points):
        r"""Axuliary method to calculate the partial effects for a given 
        variable when their covariables are fixed at mean values.

        Assumes that every variable is continuous.

        The errors will be calculated with the delta method. This method
        performs a first order taylor approximation as if the coefficients
        were a random variable.

        Let G be a function, and :math:`\mu_X` the array with the means of the 
        data variables (:math:`X=(x_1, x_2, ...)`). We expand :math:`G(X)` in
        two terms of the taylor serie, and using the covariance matrix of X
        we have:

        :math:`var(G(X)) \approx \nabla G(\mu_X)^T Cov(X) G(\mu_X).`

        In this case, X is the coefficients array and we'll use the derivatives
        w.r.t the coefficients to estimate the error.
        """

        # Making it 2d to have a more generic way of processing the coefs
        coefs = self.itexpr.coef_
        if coefs.ndim == 1:
            coefs = coefs.reshape(-1, 1).T
        
        # Partial derivatives at the means and the evaluation of each term
        at_the_means = np.zeros(
            (X.shape[1], num_points, self.itexpr.n_terms+1) )
        terms_evals  = np.zeros(
            (X.shape[1], num_points, self.itexpr.n_terms) )

        for j in range(X.shape[1]): 
            loval, hival = np.percentile(X[:, j], q=percentiles)

            # plot interval 
            Xj_range = np.linspace(loval, hival, num_points)

            for i, (fi, ti) in enumerate( self.itexpr.expr ):
                intermediary_ti = ti.copy() 

                # Let's take out the variable of interest to fix the 
                # interaction at the mean value
                intermediary_ti[j] = 0

                cov_at_means = np.prod(
                    np.power(X, intermediary_ti), axis=1).mean()

                # calculating f'(g(x))
                pi = self.tfuncs_dx[fi](np.power(Xj_range, ti[j])*cov_at_means)

                # g'(x)
                pi_partialx = ti[j]*np.power(Xj_range, ti[j]-1)*(cov_at_means)

                # Chain rule f(g(x))' = f'(g(x))*g'(x).
                at_the_means[j, :, i] = pi * pi_partialx

                # Saving the term evaluation (will be used to estimate errors)
                terms_evals[j, :, i] = \
                    self.tfuncs[fi](np.power(Xj_range, ti[j])*cov_at_means)

        # ITExpr_classifier requires special treatment to evaluate the
        # derivatives, as mentioned in BaseITExpr.gradient documentation.

        # our at_the_means array have the shape (variable, obs, term).
        # np.dot between 1d and 3d array below is a sum product over the last
        # axis of them. The result is a matrix where each line is an observation
        # and each column is the derivative w.r.t. the variable of same index.
        # The plot data is the partial effect, and each element in coefs is the
        # corresponding coefficient for each class.
        plot_data = np.array([
            np.dot(at_the_means, np.array(list(coef) + [0.0])).T
            for coef in coefs
        ])

        # If it is an ITExpr_classifier, the derivative must be calculated as
        # e^(it(x))*it'(x) / (1 + e^(it(x)))**2
        if hasattr(self.itexpr, 'classes_'):
            # calculating it(x)
            it_eval = np.array([
                np.dot(terms_evals, coef).T
                for coef in coefs])

            # adding the intercepts
            for i, itcpt in enumerate(self.itexpr.intercept_):
                it_eval[i] += itcpt

            for class_idx in range(plot_data.shape[0]):
                for var_idx in range(X.shape[1]):

                    # new values for the derivatives after the adjustments
                    plot_data[class_idx, :, var_idx] = np.divide(
                        (
                            plot_data[class_idx, :, var_idx] * \
                            np.exp(it_eval)[class_idx, :, var_idx]
                        ),
                        np.power(
                            np.exp(it_eval)[class_idx, :, var_idx] + 1.0, 2)
                    )
 
        # Calculating the errors

        if hasattr(self.itexpr, 'classes_'):
            it_eval = np.array([
                np.dot(terms_evals, coef).T
                for coef in coefs])

            for i, itcpt in enumerate(self.itexpr.intercept_):
                it_eval[i] += itcpt

            derivatives_wrt_coefs = \
                np.zeros( (X.shape[1], num_points, self.itexpr.n_terms+1) )

            for class_idx in range(plot_data.shape[0]):
                for var_idx in range(X.shape[1]):
                    for term_idx in range(self.itexpr.n_terms):
                        derivatives_wrt_coefs[var_idx, :, term_idx] = np.divide(
                            (
                                terms_evals[var_idx, :, term_idx] * \
                                np.exp(it_eval)[class_idx, :, var_idx]
                            ),
                            np.power(
                                np.exp(it_eval)[class_idx, :, var_idx] + 1.0, 2)
                        )

            variances = np.array(
                [[[(g_partial) @ cov @ (g_partial).T
                   for g_partial in derivative]
                  for derivative in derivatives_wrt_coefs]
                 for cov in self.varcovar_beta]
            )
        else:
            variances = np.array([
                [[(g_partial) @ cov @ (g_partial).T for g_partial in gradient]
            for gradient in at_the_means] for cov in self.varcovar_beta])

        # standard errors for each class
        ses = np.sqrt(variances)

        return (np.array([data.T for data in plot_data]), ses)


    def plot_partial_effects_at_means(self,
        *,
        X,
        features,
        percentiles = (5, 95),
        num_points  = 100,
        n_cols      = 3,
        target      = None,
        line_kw     = None,
        fill_kw     = None,
        ax          = None,
        show_err    = True,
        share_y     = True,
        rug         = True,
        show        = True
    ):
        """Partial effects plots for the given features, when their
        co-variables are fixed at the mean.

        .. image:: assets/images/plot_partial_effects_at_means_1.png
            :align: center
        .. image:: assets/images/plot_partial_effects_at_means_2.png
            :align: center

        Parameters
        ----------
        X : numpy.array of shape (n_samples, n_features)
            data from which we want to extract the feature importance for
            the predicted outcomes of the model.

        features : string, list[string], int, list[int] or None, default=None
            the features to generate the plots. It can be a single feature
            referred to by its label or its index or a list of features.

        percentiles : tuple of ints, default=(5, 95)
            the quartiles interval to generate the plot.

        num_points : int, default = 100
            the number of points to divide the interval when generating the 
            plots.

        n_cols : int, default=3
            number of columns to be used when creating the plot grids if
            ax is None.

        target : string, int, list[strings], list[ints] or None, default=None
            The targets to be considered when generating the plot for 
            ``ITExpr_classifier``. If the training data used strings as targets
            on ``y``, then the target must be a string of a valid class or a
            list of valid classes. If the training data was encoded as integers,
            then the target must be an int or a list of integers.
            This argument is ignored if the itexpr is an ``ITExpr_regressor``.

        line_kw : dict or None, default=None
            dictionary with keywords to be used when generating the plots.
            When set to None, then ``line_kw= {}``.

        fill_kw : dict or None, default=None
            dictionary with keywords to be used when generating the plots.
            When set to None, then ``fill_kw= {'alpha' : 0.15}``.

        ax : matplotlib.axes or list of matplotlib.axes or None, default=None
            axis to generate the plot. If none is given, then a new axis is
            created. If is a single axis, the plot will be drawn within the
            given axis. If ax is a list, then it must have the same number of
            elements in ``features``.  The new axis and figure created can be
            accessed through the ``axes_`` and ``figure_`` attributes. Notice
            that, if multiple plot functions were called, then ``axes_`` and
            ``figure_`` corresponds to the latest plot generated.

        show_err : bool, default=True
            boolean variable indicating if the standard error should be plotted.

        share_y : bool, default True
            boolean variable to specify if the axis should have the same
            interval on the y axis.

        rug : bool, default True
            whether to show the distribution as a rug. The input data will be
            divided into 10 deciles, which will be showed right above the x
            axis line.

        show :  bool, default=True
            boolean value indicating if the generated plot should be displayed
            or not.

        Raises
        ------
            ValueError
                If ``ax`` or ``target`` has invalid values.

            IndexError
                If one or more specified features are not in
                ``explainer.itexpr`` labels.

        Notes
        -----
        This plot is heavily inspired by the `Partial Dependency Plot from 
        scikit-learn
        <https://scikit-learn.org/stable/modules/partial_dependence.html>`_.
        """

        
        check_is_fitted(self)
        X = check_array(X)

        if hasattr(self.itexpr, 'classes_') and np.size(self.itexpr.classes_)>2:
            if target is None:
                target= self.itexpr.classes_
            
            target=np.array([target]).flatten()

            if not set(target).issubset(set(self.itexpr.classes_)):
                raise ValueError(f'target not in est.classes_, got {target}')

            target_idx = [list(self.itexpr.classes_).index(t) for t in target]
        else:
            # ignoring target if it is a regression problem
            target_idx = [0]

        partial_effects, standard_err = self._evaluate_partial_effects_at_means(
            X, percentiles, num_points)

        features = np.array([features]).flatten()
        if all(isinstance(n, str) for n in features):   
            features = [list(self.itexpr.labels).index(f) for f in features]

        elif all(np.isfinite(f) for f in features):
            if any(not (0 <= f < X.shape[1]) for f in features):
                raise IndexError(
                    f"Feature out of range. {features}, {X.shape[1]}")

        else:
            raise ValueError("You must give to features a list of integers or "
                             "strings corresponding to the features labels")

        if ax is None:
            fig, ax = plt.subplots()
        elif not isinstance(ax, plt.Axes):
            ax = np.asarray(ax, dtype=object)

            if ax.size != len(features):
                raise ValueError(
                    f'Expected ax to have {len(features)} axes, got {ax.size}')
        
        default_fill_kw = {'alpha' : 0.15}
        if fill_kw is None:
            fill_kw = default_fill_kw
        else:
            fill_kw = {**default_fill_kw, **fill_kw}

        default_line_kw = {}
        if line_kw is None:
            line_kw = default_line_kw
        else:
            line_kw = {**default_line_kw, **line_kw}
    
        # Creating subplots if ax is a single axis
        if isinstance(ax, plt.Axes):
            n_cols = min(n_cols, len(features))
            n_rows = int(np.ceil(len(features) / float(n_cols)))
            
            ax.set_axis_off()

            self.figure_ = ax.figure
            self.axes_ = np.empty((n_rows, n_cols), dtype=object)

            axes_ravel = self.axes_.ravel()

            gs = GridSpecFromSubplotSpec(n_rows, n_cols,
                                         subplot_spec=ax.get_subplotspec())

            for i, spec in zip(range(len(features)), gs):
                axes_ravel[i] = self.figure_.add_subplot(spec)

        else:
            if ax.ndim == 2:
                n_cols = ax.shape[1]
            else:
                n_cols = None

            self.axes_   = ax
            self.figure_ = ax.ravel()[0].figure

        ymin = np.min(partial_effects[target_idx][:, features])
        ymax = np.max(partial_effects[target_idx][:, features])
            
        # finally generating the plots
        for plot_idx, (axi, feature_idx) in enumerate(
            zip(self.axes_.ravel(), features)
        ):
            loval, hival = np.percentile(X[:, feature_idx], q=percentiles)
            Xj_range = np.linspace(loval, hival, num_points)

            for target_i in target_idx:
                axi.plot(Xj_range, partial_effects[target_i, feature_idx, :],
                label=(self.itexpr.classes_[target_i]
                if hasattr(self.itexpr, 'classes_') else None), **line_kw)

                if show_err:
                    low_bound = [d+e for d, e in zip(
                            partial_effects[target_i, feature_idx, :], \
                            standard_err[target_i, feature_idx, :])]

                    upper_bound = [d-e for d, e in zip(
                            partial_effects[target_i, feature_idx, :], \
                            standard_err[target_i, feature_idx, :])]
                    
                    axi.fill_between(
                        Xj_range, low_bound, upper_bound, **fill_kw)
                
            if self.itexpr.labels is not None and len(self.itexpr.labels)>0:
                axi.set_xlabel(self.itexpr.labels[feature_idx])
            else:
                axi.set_xlabel(feature_idx)

            if (n_cols is None and plot_idx == 0) or \
               (n_cols is not None and plot_idx % n_cols == 0):
                if not axi.get_ylabel():
                    axi.set_ylabel("partial effect\nat the means")
            else:
                if share_y:
                    axi.set_yticklabels([])
            
            if share_y:
                margin = (ymax - ymin)*0.05
                axi.set_ylim( (ymin - margin, ymax + margin) )
            else:
                ymin = np.min(partial_effects[:, feature_idx])
                ymax = np.max(partial_effects[:, feature_idx])

                # Handling constant partial effects
                if ymin == ymax:
                    axi.set_ylim((ymin*0.99, ymax*1.01))
                    ymin = ymin * 0.991
                
            if rug==True:
                # Dividing the interval into 10 deciles and plotting above the 
                # x axis
                deciles = mquantiles(
                    X[:, feature_idx],
                    prob=np.arange(0.1, 1.1, 0.1),limit=(loval, hival))            
                
                trans = transforms.blended_transform_factory(
                    axi.transData, axi.transAxes)
                
                axi.vlines(
                    deciles,
                    0,
                    0.05,
                    transform=trans,
                    color="k",
                )

            if hasattr(self.itexpr, 'classes_'):
                axi.legend()

        if show:
            plt.show()


    def plot_normalized_partial_effects(self, *,
        num_points         = 20,
        grouping_threshold = 0.05,
        stack_kw           = None,
        ax                 = None,
        show               = True
    ):
        """Partial effects plots, separing the training data into discrete
        intervals.

        First, the output interval is discretized. Then, for each interval,
        the partial effect of the sample data that predict values within the 
        interval is calculated. Finally, they are normalized in order to 
        make the total contribution by 100%.

        .. image:: assets/images/plot_normalized_partial_effects_1.png
            :align: center

        Parameters
        ----------

        num_points : int, default = 20
            the number of points to divide the interval when generating the 
            plots. The value must be equal or greater than 2. Large values can
            end up creating intervals where no predictions in the training data
            lies within resulting in points that are not evenly distributed, or
            in a noisy plot. The adequate number of points can be different for
            each dataset.
            This parameter is ignored if ``itexpr == ITExpr_classifier``.

        grouping_threshold : float, default = 0.05
            The features will be iterated in order of importance, from the
            smallest to the highest importance, and a group of the smallest
            features that sum up less than the given percentage of importance
            will be grouped to reduce the plot information.
            To disable the creation of the group, set ``grouping_threshold=0``.

        stack_kw : dict or None, default=None
            dictionary with keywords to be used when generating the plots.
            When set to None, then stack_kw is ``{'baseline': 'zero',
            'labels': labels, 'edgecolor': 'k', 'alpha': 0.75}``.

        ax : matplotlib.axes or None, default=None
            axis to generate the plot. If none is given, then a new axis is
            created. if is a single axis, the plot will be drawn within the
            given axis.  The new axis and figure created can be accessed through
            the ``axes_`` and ``figure_`` attributes. Notice that, if multiple
            plot functions were called, then ``axes_`` and ``figure_``
            corresponds to the latest plot generated.

        show : bool, default=True
            boolean value indicating if the generated plot should be displayed
            or not.

        Raises
        ------
            ValueError
                If ``ax`` has invalid values.

        Notes
        -----
        This plot was inspired by the relative feature contribution plot
        reported in the paper:
        "R. M. Filho, A. Lacerda and G. L. Pappa, "Explaining Symbolic
        Regression Predictions," 2020 IEEE Congress on Evolutionary Computation
        (CEC)".
        """

        check_is_fitted(self)

        if ax is None:
            self.figure_, self.axes_ = plt.subplots()
        elif not isinstance(ax, plt.Axes):
                raise ValueError(
                    f'Expected ax to have 1 axes, got {np.asarray(ax).size}')
        else:
            self.figure_ = ax.figure
            self.axes_   = ax
        
        predictions = self.itexpr.predict(self.X_)

        # Handling classification and regression separately
        if hasattr(self.itexpr, 'classes_'):
            num_points = len(self.itexpr.classes_)

            global_importances  = np.zeros( (self.X_.shape[1], num_points) )

            # We'll save the intervals with no predictions to exclude
            invalid_importances = np.zeros(num_points, dtype=np.int32)

            for i, c in enumerate(self.itexpr.classes_):
                mask = np.where(predictions == c)[0]
                
                if len(mask)==0:
                    invalid_importances[i] = 1
                    continue

                feature_importances = self.average_partial_effects(
                    self.X_[mask, :])[i]

                feature_importances /= np.sum(feature_importances)/100
                
                global_importances[:, i] = feature_importances

            valid_importance_mask = np.where(invalid_importances != 1)[0]
            global_importances = global_importances[:, valid_importance_mask]
            x_axis = np.arange(len(valid_importance_mask))
        else:
            x_axis = np.linspace(
                np.min(predictions), np.max(predictions), num_points+1)

            global_importances = np.zeros( (self.X_.shape[1], num_points) )

            invalid_importances = np.zeros(num_points, dtype=np.int32)

            for i in range(num_points):
                mask = np.where(
                    (x_axis[i] <= predictions) &
                    (predictions < x_axis[i+1])
                )[0]

                if len(mask)==0:
                    invalid_importances[i] = 1
                    continue

                feature_importances = self.average_partial_effects(
                    self.X_[mask, :])                
                
                feature_importances /= np.sum(feature_importances)/100
                
                global_importances[:, i] = feature_importances

            valid_importance_mask = np.where(invalid_importances != 1)[0]
            global_importances = global_importances[:, valid_importance_mask]
            x_axis = (x_axis[:-1])[valid_importance_mask]

        # It must be a np array so we can use lists as indexes to sort labels
        labels = np.array(self.itexpr.labels)

        # order is going from the highest to the smallest important features
        order = np.argsort(-np.sum(global_importances, axis=1))

        # grouping the features that have small contribution
        others = np.zeros_like(global_importances[0])
        features_in_others = 0
        thresold_area = grouping_threshold*100*num_points

        for i in order[::-1]:
            if np.sum(others + global_importances[i, :]) < thresold_area:
                others += global_importances[i, :]
                features_in_others += 1

        # Greater than one because a group of one is not a group
        if features_in_others > 1:
            sorted = order[:len(order)-(features_in_others)]
            global_importances = np.vstack((
                global_importances[sorted],
                others
            ))

            labels = np.hstack((
                labels[sorted],
                [f'Other features ({features_in_others})']
            ))

        else:
            global_importances = global_importances[order]
            labels = labels[order]

        default_kw = {
            'baseline'  : 'zero',
            'labels'    : labels,
            'edgecolor' : 'k',
            'alpha'     : 0.75}

        if stack_kw is None:
            stack_kw = default_kw
        else:
            stack_kw = {**default_kw, **stack_kw}

        self.axes_.stackplot(
            x_axis,
            *global_importances,
            **stack_kw
        )

        self.axes_.set_xlabel("Model prediction")
        self.axes_.set_ylabel("Relative importance (%)")

        if hasattr(self.itexpr, 'classes_'):
            self.axes_.set_xticks(x_axis)
            self.axes_.set_xticklabels(
                self.itexpr.classes_[valid_importance_mask],
                rotation=30, ha='right')
            self.axes_.set_xlim( (0, len(x_axis)-1) )
        else:
            self.axes_.set_xlim( (x_axis[0], x_axis[-1]) )

        handles, labels = self.axes_.get_legend_handles_labels()
        self.axes_.legend(
            reversed(handles), reversed(labels),
            bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.25)

        self.axes_.set_ylim( (0, 100) )

        if show:
            plt.show()