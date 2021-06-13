# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.2
# Last modified: 06-09-2021 by Guilherme Aldeia


"""ITExpr_texifier class.
"""


import numpy as np


class ITExpr_texifier:
    """class containing static methods to create LaTeX representations of the
    expression.
    """
    
    @staticmethod
    def _term_frac(term):
        """Method that takes an array of exponents and return a string
        containing a latex frac with placeholders on the name of the variables.
        """

        fi, ti = term

        if np.all(ti == 0): 
            return f'{fi}(1.0)'

        num = r' \cdot '.join(
            [fr'var_placeholder_{i}^{ {e} }' for i, e in enumerate(ti) if e>0]
        ).replace(r'^{1}', '')

        den = r' \cdot '.join(
            [fr'var_placeholder_{i}^{ {-e} }' for i, e in enumerate(ti) if e<0]
        ).replace(r'^{1}', '')

        if len(den)==0:
            return f'{fi}({num})'
        elif len(num)==0:
            return f'{fi}(' + r'\frac{1}{' + den + r'})'
        else:
            return f'{fi}(' + r'\frac{' + num + '}{' + den + r'})'


    @staticmethod
    def to_latex(
        itexpr, term_separator=' + ', term_wrapper=None):
        r"""Static method that takes an instance of an ``ITExpr`` and returns
        a latex representation of the expression.

        Parameters
        ----------
        itexpr : ITExpr_classifier or ITExpr_regressor
            an instance of itexpr to be texified.

        term_separator : string, default=' + '
            string to be used to concatenate each term.

        term_wrapper : None or Callable, default=None
            a function that takes two arguments: ``i`` the index of the term
            and ``term`` the term itself, and returns a string. This can be
            used to add special formatting to the terms. Examples are:

            - ``lambda i, term: r'\underbrace{'+term+r'}
              _{\text{term '+str(i)+'}}'`` to add a underbracket with the 
              index of the term;
            - ``lambda i, term: term`` to do nothing;

            When set to None, then the latter expression will be used.

        Returns
        -------
        itexpr_latex : string
            latex expression representing the given itexpr.
        """

        if term_wrapper== None:
            term_wrapper = lambda i, term: term

        latex_terms  = []

        for i, term in enumerate(itexpr.expr):
            latex_terms.append(term_wrapper(
                i, 
                f'\\beta_{i} \\cdot {ITExpr_texifier._term_frac(term)}'
            ))

        str_it = term_separator.join(latex_terms)

        if len(itexpr.labels)>0:
            for i, l in enumerate(itexpr.labels):
                str_it = str_it.replace(f"var_placeholder_{i}", l)
        else:
            str_it = str_it.replace(f"var_placeholder_", "x")

        return str_it + ' + I_0'


    @staticmethod
    def derivatives_to_latex(
        itexpr, term_separator=' + ', term_wrapper=None):
        r"""Static method that takes an instance of an ``ITExpr`` and returns
        a list containing a latex representation of each partial derivative
        of the expression.

        Parameters
        ----------
        itexpr : ITExpr_classifier or ITExpr_regressor
            an instance of itexpr to be texified.

        term_separator : string, default=' + '
            string to be used to concatenate each term.

        term_wrapper : None or Callable, default=None
            a function that takes two arguments: ``i`` the index of the term
            and ``term`` the term itself, and returns a string. This can be
            used to add special formatations to the terms. Examples are:

            - ``lambda i, term: r'\underbrace{'+term+r'}
              _{\text{term '+str(i)+'}}'`` to add a underbracket with the 
              index of the term;
            - ``lambda i, term: term`` to do nothing;

            When set to None, then the latter expression will be used.

        Returns
        -------
        itexpr_latexs : list[string]
            list of strings, where each string is a latex representation
            of the partial derivative of the given itexpr.
        """
        
        if term_wrapper== None:
            term_wrapper = lambda i, term: term

        # number of variables is the length of any exponent array
        n_vars = len(itexpr.expr[0][1])

        latex_derivatives = []
        for j in range(n_vars):
            latex_terms  = []
            for i, term in enumerate(itexpr.expr):
                fi, ti = term
                term_dx = ti.copy()
                
                term_dx[j] -= 1

                # Chain rule
                inner_dx = ITExpr_texifier._term_frac( ('', term_dx) )
                outer_dx = ITExpr_texifier._term_frac( (f'{fi}\'', ti))

                if ti[j] != 0:
                    latex_terms.append(term_wrapper(
                        i, 
                        f'{ti[j]}\\beta_{i} \\cdot {outer_dx}{inner_dx}'
                    ))

            if len(latex_terms) > 0:
                str_it = term_separator.join(latex_terms)
            else:
                str_it = '0.0'

            if len(itexpr.labels)>0:
                for i, l in enumerate(itexpr.labels):
                    str_it = str_it.replace(f"var_placeholder_{i}", l)
            else:
                str_it = str_it.replace(f"var_placeholder_", "x")

            latex_derivatives.append(str_it)

        return latex_derivatives