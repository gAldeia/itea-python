# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.3
# Last modified: 24-11-2021 by Guilherme Aldeia


"""ITEA_summarizer class.
"""


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from pylatex         import Document, Section, Command, Figure, Table
from pylatex.utils   import NoEscape, make_temp_dir, rm_temp_dir
from pylatex.package import Package

from itea.inspection import ITExpr_inspector, ITExpr_explainer, ITExpr_texifier
from matplotlib.gridspec      import GridSpecFromSubplotSpec 

from sklearn.utils.validation import check_array, check_is_fitted


class ITEA_summarizer:
    """Class to automatically generate a pdf file reporting
    several interpretability plots for the expression.
    """

    def __init__(self, *, itea):
        """Constructor method.

        Parameters
        ----------
        itea : ITEA_classifier or ITEA_regressor
            fitted instance of an ``ITEA`` class to be summarized.
        """

        self.itea = itea

        self.packages = {
            'geometry'    : {
                "paperwidth"  : "16cm",
                "paperheight" : "12cm",
                "tmargin"     : "1.75cm",
                "lmargin"     : "1cm",
                "rmargin"     : "1cm",
                "bmargin"     : "1.5cm",
            },
            'fontenc'      : ['T1'],
            'babel'       : 'english',
            'datetime'    : [],
            'grffile'     : [],
            'booktabs'    : [],
            'amsfonts'    : [],
            'amssymb'     : [],
            'amsmath'     : [],
            'amsthm'      : [],
            'breqn'       : [],
            'fancyhdr'    : [],
            'indentfirst' : [],
            'float'       : [],
        }
    
    
    def _report_frontpage(self, doc, save_path):
        """Private method to create the frontpage of the report
        """

        for k, v in self.packages.items():
            doc.preamble.append(Package(k, v))

        doc.preamble.append(Command('title', 'ITEA automatic report'))
        doc.preamble.append(
            Command('author', NoEscape(r'\textit{ITEA\_summarizer}')))
        doc.preamble.append(Command('date', NoEscape(r'\today, \currenttime')))
        doc.preamble.append(Command('pagestyle', 'fancy'))
        
        doc.append(NoEscape(r"""\maketitle \vfill            
            
            Automatic report created by \textit{ITEA\_summarizer} package.
            This report makes usage of several methods to automatically inspect
            and explain the final expression found in the evolutionary process
            performed by the ITEA algorithm.

            \vfill \pagebreak"""))


    def _report_pre_execution(self, doc, save_path):
        """Creates all pages with information related to pre-execution
        of the algorithm (such as hyperparameters).
        """
    
        # Header and footer
        doc.append(NoEscape(r"""
            \lhead{Pre-execution --- ITEA automatic report}
            \chead{}
            \rhead{\today, \currenttime}
            
            \lfoot{}
            \cfoot{}
            \rfoot{\thepage\ | \pageref{LastPage}}"""))

        # reporting descriptive statistics for the variables (maximum 5 to 
        # avoid overfull tables)
        with doc.create(Section(
            NoEscape('Descriptive statistics of the data'), numbering=False)):
            
            # Retrieving the feature importances and selecting at most 5
            feature_importances = self.itea.bestsol_.feature_importances_
            
            order = np.argsort(-np.sum(feature_importances, axis=0))
            
            # get the most relevant features
            selected = order[:np.minimum(5, len(order))]

            doc.append(NoEscape(f"""
                Reporting descriptive statistics for {np.minimum(5, len(order))}
                (from a total of {len(order)}) features contained on the
                training data. The features were selected based on the absolute
                final importance."""))

            df_summary = pd.DataFrame(
                np.array(self.X_)[:, selected],
                columns=np.array(self.itea.labels)[selected]
            )
            with doc.create(Table(position='H')) as table:
                table.append(Command('centering'))
                table.append(Command('footnotesize'))
                table.append(NoEscape(
                    df_summary.describe().to_latex(escape=True)))
            
            doc.append(NoEscape(r"\vfill \pagebreak"))

        # reporting the hyper parameters
        with doc.create(Section(
            NoEscape('Algorithm Hyper-parameters'), numbering=False)):
            
            doc.append(NoEscape(r"""
                The following hyperparameters were used to execute the
                algorithm. If the random\_state parameter was set to an 
                integer value (or a numpy randomState instance was given), then
                it is possible to repeat the exact execution by using the same
                training data and the parameters listed below."""))

            tfuncs_names = self.itea.tfuncs.keys()

            doc.append(NoEscape(
                r"{\footnotesize \begin{verbatim}" +
                            
                # almost all hyperparameters
                '\n'.join([f"    {k} : {v}"
                    for (k, v) in self.itea.get_params().items()
                    if 'funcs' not in k and k!= 'labels']) + 

                # displaying the transformation function keys
                f"\n    tfuncs : [{', '.join([k for k in tfuncs_names])}]" +
                            
                r"\end{verbatim} } \vfill \pagebreak"
            ))


    def _report_execution(self, doc, save_path):
        """Creates pages with information about the ITEA and the ITExpr.
        This section of the report does not include the post hoc explanations.
        """

        # Auxiliary function to use in texifier
        def term_wrapper_f(i, term):
            return r'\underbrace{' + term + r'}_{\text{term ' + str(i) + '}}'

        doc.append(NoEscape(r"""
            \lhead{Execution --- ITEA automatic report}
            \chead{}
            \rhead{\today, \currenttime}
            
            \lfoot{}
            \cfoot{}
            \rfoot{\thepage\ | \pageref{LastPage}}
        """))

        # Convergence
        with doc.create(Section(
            NoEscape("Evolution convergence"), numbering=False)):

            doc.append(NoEscape(f"""
                The algorithm took {round(self.itea.exectime_, 3)} seconds to
                completely run. Below are the plots for the average fitness
                of the population and the best individual fitness for each
                generation.""" + 
                r"\vfill"))

            fig, axs = plt.subplots(1, 1, figsize=(8, 3))

            self.plot_convergence(
                data='fitness',
                ax=axs,
                show=False
            )

            plt.tight_layout()
            plt.savefig(f"{save_path}/fitness_convergence.pdf")
            plt.close()

            with doc.create(Figure(position='H')) as figure_plot:
                figure_plot.add_image(
                    f"{save_path}/fitness_convergence.pdf",
                    width=NoEscape(r'0.8\textwidth')
            )
                
            doc.append(NoEscape(r"\vfill \pagebreak"))

        # Final expression descriptions
        with doc.create(Section(
            NoEscape('Best expression'), numbering=False)):

            if hasattr(self.itea, 'classes_'):
                type_itexpr = 'classifier'
            else:
                type_itexpr = 'regressor'

            doc.append(NoEscape(r"""
                The best expression corresponds to the expression with
                the best fitness on the last generation before the evolution
                ends. Not necessarily it will be the simpliest or the global
                optimum expression of the evoution. """ + 
            
                f"""The final expression is a {type_itexpr} with a fitness of
                {round(self.itea.fitness_, 5)}, and the number of IT terms is
                {self.itea.bestsol_.n_terms}. Below is an LaTeX representation
                of the expression:
                
                """ + 

                r"\vfill {\small \begin{dmath}" + 
                
                NoEscape("ITExpr = " + ITExpr_texifier.to_latex(
                    self.itea.bestsol_,
                    term_wrapper = term_wrapper_f
                )) + 
                
                r"\end{dmath} } \vfill \pagebreak"))

        # inspector statistics
        with doc.create(Section(
            NoEscape('Best expression metrics'), numbering=False)):

            doc.append(NoEscape(r"""On the next page is reported a table
            containing the coefficients for the previous expression, as well as
            some metrics calculated for each term individually:
            
            \begin{itemize}
            \item \textbf{coef:} coefficient of each term (or coefficients,
                  if the itexpr is an instance of ITExpr_classifier);

            \item \textbf{coef stderr:} the standard error of the coefficients;

            \item \textbf{disentang.:} mean pairwise disentanglement between
                  each term when compared with the others;

            \item \textbf{M.I.:} mean continuous mutual information between
                  each term when compared with the others;

            \item \textbf{pred. var.:} variance of the predicted outcomes for
                  each term when predicting the training data.
            \end{itemize}

             \vfill \pagebreak"""))

            statistics = pd.DataFrame(self.inspector_.terms_analysis())
            statistics = statistics.drop(columns='strengths')
            statistics.columns = ['coef', 'func', 'coef stderr',
                                  'disentang.', 'M.I.', 'pred. var.']
            statistics = statistics.set_index(
                'term ' + statistics.index.astype(str))

            with doc.create(Table(position='H')) as table:
                table.append(Command('centering'))
                table.append(Command('footnotesize'))
                table.append(NoEscape(statistics.to_latex(escape=True)))
            
            doc.append(NoEscape(r"\vfill \pagebreak"))

        # Partial derivatives
        with doc.create(Section(
            NoEscape('Partial derivatives'), numbering=False)):

            derivatives_latex = ITExpr_texifier.derivatives_to_latex(
                self.itea.bestsol_,
                term_wrapper = term_wrapper_f
            )

            out = r"{\footnotesize"
            
            for l, d in zip(self.itea.labels, derivatives_latex):
                out += (
                    r"\begin{dmath}" + 
                    r"\frac{\partial}{\partial " + str(l) + "} ITExpr = " + d + 
                    r"\end{dmath}"
                )
            
            doc.append(NoEscape(out + r"} \vfill \pagebreak"))


    def _report_post_execution(self, doc, save_path, importance_methods):
        """Post hoc interpretations of the ITExpr. Several plots will be
        generated.
        """
        if importance_methods is None:
            importance_methods = 'pe'

        importance_methods=np.array([importance_methods]).flatten()

        if not set(importance_methods).issubset(set(['pe', 'ig', 'shapley'])):
            raise ValueError(f'importance_methods not in est.classes_, ', 
                             f'got {importance_methods}')

        explainer_headers = {
            'pe': r'Global importances with \textit{Average partial Effects}',
            'ig': r'Global importances with \textit{Integrated Gradients}',
            'shapley': r'Global importances with \textit{Shapley Values}',
        }

        explainers_descriptions = {
            'pe' : r"""
                Feature importances with Average Partial Effects. This method
                attributes the importance to the i-th variable by calculating
                the average of the partial derivative w.r.t. i, evaluated for
                all data in the training set.

                \vfill""",

            'ig' : r"""
                Feature importance using the Average Integrated Gradients
                importances. The idea is to calculate a local
                importance score for a feature $i$ by evaluating the integral of
                the models' gradients $\frac{\partial f}{\partial x_i}$ along a
                straight line between one baseline and the specific point.
            
                \vfill""",

            'shapley' : r"""
                Feature importance with the average approximation of the 
                Shapley values. The shapley values are based on coalition game
                theory, where players contribute differently to the team. The
                Shapley value is the total contribution of the player, and
                represents the overall contribution of the player.
            
                \vfill"""
        }

        explainer_colors = {
            'pe' : 'green',
            'ig' : 'blue',
            'shapley' : 'red'
        }

        # One image for each explainer
        for importance_method in importance_methods:
            
            doc.append(NoEscape(r"""
                \lhead{post-execution --- ITEA automatic report}
                \chead{}
                \rhead{\today, \currenttime}
                
                \lfoot{}
                \cfoot{}
                \rfoot{\thepage\ | \pageref{LastPage}}
            """))

            # Average partial effects
            with doc.create(Section(
                NoEscape(explainer_headers[importance_method]), numbering=False)):   

                doc.append(NoEscape(explainers_descriptions[importance_method]))

                fig, axs = plt.subplots(1, 1, figsize=(8, 4))

                self.explainer_.plot_feature_importances(
                    X = self.X_,
                    ax = axs,
                    importance_method  = importance_method,
                    grouping_threshold = 0.05,
                    target = None,
                    barh_kw = {
                        'edgecolor' : 'k',
                        'alpha' : 0.8,
                        'facecolor' : explainer_colors[importance_method]},
                    show = False
                )

                plt.tight_layout()
                plt.savefig(f"{save_path}/{importance_method}.pdf")
                plt.close()
                
                with doc.create(Figure(position='H')) as figure_plot:
                    figure_plot.add_image(
                        f"{save_path}/{importance_method}.pdf",
                        width=NoEscape(r'0.8\textwidth')
                )
                    
                doc.append(NoEscape(r"\vfill \pagebreak"))

        # Normalized partial effects
        with doc.create(Section(
            NoEscape(r'\textit{Normalized partial Effects}'), numbering=False)):   

            doc.append(NoEscape(r"""
                Feature importances with Normalized Partial Effects. 
                To create this plot, first, the output interval is discretized.
                Then, for each interval, the partial effect of all samples
                in the training set that results in a prediction within the
                interval are calculated. Finally, they are normalized in
                order to make the total contribution by 100\%.

                \vfill"""))

            fig, axs = plt.subplots(1, 1, figsize=(8, 4))

            self.explainer_.plot_normalized_partial_effects(
                ax = axs,
                grouping_threshold = 0.05,
                show = False
            )

            plt.tight_layout()
            plt.savefig(f"{save_path}/normalized_partial_effects.pdf")
            plt.close()
            
            with doc.create(Figure(position='H')) as figure_plot:
                figure_plot.add_image(
                    f"{save_path}/normalized_partial_effects.pdf",
                    width=NoEscape(r'0.8\textwidth')
            )
                
            doc.append(NoEscape(r"\vfill \pagebreak"))

        # Partial effects at the means
        with doc.create(Section(
            NoEscape(r'\textit{Partial Effects at the Means}'),
            numbering=False
        )):   

            doc.append(NoEscape(r"""
                Partial Effects plots created by fixing the co-variables at
                the means and evaluating the model's output when only one
                variable changes. For simplicity, at most 5 variables are
                selected to create the plot (the 5 most important variables
                considering their Average Partial Effects).

                \vfill"""))

            fig, axs = plt.subplots(1, 1, figsize=(9, 3))

            feature_importances = self.itea.bestsol_.feature_importances_
            
            order = np.argsort(-np.sum(feature_importances, axis=0))
            
            # get the most relevant features
            selected = order[:np.minimum(5, len(order))]

            self.explainer_.plot_partial_effects_at_means(
                X=self.X_,
                features=selected,
                ax=axs,
                n_cols=5,
                num_points=100,
                share_y=True,
                show_err=True,
                show=False
            )

            plt.tight_layout()
            plt.savefig(f"{save_path}/partial_effets_at_means.pdf")
            plt.close()
            
            with doc.create(Figure(position='H')) as figure_plot:
                figure_plot.add_image(
                    f"{save_path}/partial_effets_at_means.pdf",
                    width=NoEscape(r'\textwidth')
            )
                
            doc.append(NoEscape(r"\vfill \pagebreak"))


    def fit(self, X, y):
        """Fit method to store the data used in the training of the given
        itea instance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            data used to train the itexpr model.

        y : array-like of shape (n_samples, )
            target data used to train the itexpr model.

        Returns
        -------
        self : ITEA_summarizer
        """

        X = check_array(X)
        
        self.X_ = X
        self.y_ = y

        self.inspector_ = ITExpr_inspector(
            itexpr=self.itea.bestsol_, tfuncs=self.itea.tfuncs
        ).fit(self.X_, self.y_)

        self.explainer_ = ITExpr_explainer(
            itexpr=self.itea.bestsol_, tfuncs=self.itea.tfuncs,
            tfuncs_dx=self.itea.tfuncs_dx
        ).fit(X, y)

        return self


    def plot_convergence(self,
        *,
        data     = None,
        n_cols   = 1,
        line_kw  = None,
        fill_kw  = None,
        ax       = None,
        show_err = True,
        show     = True
    ):
        """Plot of information about the ``itea`` evolutionary process.
        This function is intended to help visualize the information on the
        ``itea.convergence_`` dictionary.

        .. image:: assets/images/plot_convergence_1.png
            :align: center

        Parameters
        ----------
        data : string, list of string, or None, default=None
            the convergence information to generate the plots. It can be a
            single string or a list with strings in
            ``['fitness', 'n_terms', 'complexity']``. If set to none, then
            the whole list of strings will be used.

        n_cols : int, default=1
            number of columns to be used when creating the plot grids if ax is
            None.

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
            elements in ``data``.

        show_err : bool, default=True
            boolean variable indicating if the standard error should be plotted.

        show :  bool, default=True
            boolean value indicating if the generated plot should be displayed
            or not.

        Raises
        ------
        ValueError
                If ``ax`` or ``data`` has invalid values.
        """
        
        check_is_fitted(self)

        if data is None:
            data = ['fitness', 'n_terms', 'complexity']
            
        data = np.array([data]).flatten()
        if not set(data).issubset(['fitness', 'n_terms', 'complexity']):
            raise ValueError("Data must be one string or a list containing "
                             "one or more of the following strings: "
                             "'fitness', 'n_terms', 'complexity'")

        if ax is None:
            fig, ax = plt.subplots()
        elif not isinstance(ax, plt.Axes):
            ax = np.asarray(ax, dtype=object)

            if ax.size != len(data):
                raise ValueError(
                    f"Expected ax to have {len(data)} axes, got {ax.size}. "
                    "The number of axes must be equal to the number of "
                    "values in `data` (or 1 if data is a string).")

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
            n_cols = min(n_cols, len(data))
            n_rows = int(np.ceil(len(data) / float(n_cols)))
            
            ax.set_axis_off()

            self.figure_ = ax.figure
            self.axes_ = np.empty((n_rows, n_cols), dtype=object)

            axes_ravel = self.axes_.ravel()

            gs = GridSpecFromSubplotSpec(n_rows, n_cols,
                                         subplot_spec=ax.get_subplotspec())

            for i, spec in zip(range(len(data)), gs):
                axes_ravel[i] = self.figure_.add_subplot(spec)

        else:
            if ax.ndim == 2:
                n_cols = ax.shape[1]
            else:
                n_cols = None

            self.axes_   = ax
            self.figure_ = ax.ravel()[0].figure
    
        best_of_generation = 'max' if self.itea._greater_is_better else 'min'        
        gens = range(len(self.itea.convergence_['fitness']['mean']))

        for (axi, d) in zip(self.axes_.ravel(), data):
            axi.plot(
                gens,
                self.itea.convergence_[d]['mean'],
                label='mean',
                **line_kw
            )

            if d == 'fitness':
                # only the fitness can have either higher or lower values
                # as the best for each generation. All other data always
                # have smaller values as better values.
                axi.plot(
                    gens,
                    self.itea.convergence_[d][best_of_generation],
                    label=f"best ({best_of_generation}imum)",
                    **line_kw
                )
            else:
                axi.plot(
                    gens,
                    self.itea.convergence_[d]['min'],
                    label=f"best (minimum)",
                    **line_kw
                )

            if show_err:    
                low_bound = [y+std for y, std in zip(
                    self.itea.convergence_[d]['mean'],
                    self.itea.convergence_[d]['std'])]

                upper_bound = [y-std for y, std in zip(
                    self.itea.convergence_[d]['mean'],
                    self.itea.convergence_[d]['std'])]

                axi.fill_between(
                    gens, low_bound, upper_bound, **fill_kw)

            axi.set_title(d)
            axi.legend()
            axi.set_xlabel("generation")

        if show:
            plt.show()


    def autoreport(self,
        importance_methods=None, save_path='.',name_suffix='', use_temp_folder=True):      
        """automatically generate a pdf using the methods implemented in
        ``ITExpr_inspector``, ``ITExpr_explainer``, and ``ITExpr_texifier``.

        The idea is to simplify the generation of the plots and tables,
        removing from the user the need to understand, instantiate the classes
        and call the plots functions.
        
        All explanations are generated with the training data, and every
        item in the report can be obtained manually by using the
        ``ITExpr_inspector``, ``ITExpr_explainer``, and ``ITExpr_texifier``.

        This method makes usage of the ``PyLaTeX`` package and requires a 
        visible latex installation to work properly.

        The .tex file used to generate the pdf will also be saved on the
        designed path.

        You can download one example of report
        :download:`by clicking here </assets/files/Report.pdf>`.

        Parameters
        ----------
        importance_methods : string or list[strings] or None, default=None
            Feature importance method(s) used to generate explanations in
            the report. Must be one of the possible explainers implemented 
            (``['pe', 'ig', 'shapley']``) or a list containing one or more
            of the explainers. If None, then ``'pe'`` will be used. The report
            will contain one page for each method specified here.

        save_path : string, default='.'
            path to save the pdf report. The file will be saved as "Report.pdf",
            unless a ``name_suffix`` is provided.
            A Te

        name_suffix : string, default=""
            suffix to add in the name of the report. 

        use_temp_folder : boolean, defaut=True
            specifies if a temporary folder should be used to save the plots
            during the creation of the report. If false, then the plots will
            be saved on the ``save_path``.
        """

        check_is_fitted(self)

        # Creating the doc and title page
        doc = Document('Report')

        if use_temp_folder:
            temp_path = make_temp_dir()
        else:
            temp_path = save_path

        self._report_frontpage(doc, temp_path)
        self._report_pre_execution(doc, temp_path)
        self._report_execution(doc, temp_path)
        self._report_post_execution(doc, temp_path, importance_methods)

        doc.generate_pdf(f'{save_path}/Report{name_suffix}', clean_tex=False)

        if use_temp_folder:
            rm_temp_dir()
