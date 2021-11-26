# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.9
# Last modified: 09-03-2021 by Guilherme Aldeia


"""Base class to be inherited for classification and regression tasks."""


import time
import warnings

import numpy  as np

from sklearn.base             import BaseEstimator
from sklearn.utils.validation import check_random_state

from itea._manipulators.generator import uniform
from itea._manipulators.mutation  import mutate_individual
from itea._manipulators.sanitizer import sanitize
from itea.inspection              import ITExpr_explainer

import itea._manipulators.simplifier as simplifiers


class BaseITEA(BaseEstimator):
    """Base class to be inherited for classification and regression tasks.

    This class implements argument checks and generic evolutionary methods
    (population initialization, selection, mutation, and evolution), along with
    three virtual methods to be implemented.
    
    Ideally, this class should never be instantiated, only its derivations.

    Its derivations will be scikit estimators and can be used in many scikit
    methods such as grid-search or pipelines.

    Every argument is a named argument. The list of arguments includes 
    everything that an ``ITExpr`` class needs to be instantiated.

    All arguments have a default value. In this configuration, the
    evolutionary process will search only for polynomials.
    """
    
    def __init__(self, *,
        gens            = 100,
        popsize         = 100,
        expolim         = (-2, 2),
        max_terms       = 5,
        simplify_method = None,
        random_state    = None,
        verbose         = None,
        labels          = [],
        tfuncs          = {'id': lambda x: x},
        tfuncs_dx       = None,
        predictor_kw    = None  
    ):
        """Constructor method.

        Parameters
        ----------

        gens : int, default=100
            number of generations of the evolutionary process. The
            algorithm does not implement an early stop mechanism, so
            it is guaranteed that the algorithm will perform the exact
            number of generations.

        popsize : int, default=100
            population size, consistent through each generation.

        expolim : tuple (int, int), default = (-2, 2)
            tuple containing two integers, specifying the bounds
            of exponents that can be explored through the evolution.

        max_terms : int, default=5
            the max number of IT terms allowed.

        simplify_method : string or None, default=None
            String with the name of the simplification method to be used
            before fitting expressions through the evolutionary process.
            When set to None, the simplification step is disabled.

            Simplification can impact performance. To be simplified, the
            expression must be previously fitted. After the simplification, if
            the expression was changed, it should be fitted again to better
            adjust the coefficients and intercept to the new IT expressions'
            structure.

        random_state : int, None or numpy.random_state, default=None
            int or numpy random state. Use this argument
            to have reproducible results across different
            executions. When None, a random state instance
            will be created and used and can be accessed
            by ``itea.random_state``.

        verbose : int, None or False, default=None
            specify if the algorithm should perform the evolution
            silently or if it should print information through the
            process. When verbose is None, False, or 0, the algorithm
            will not print any information. If verbose is an integer
            ``n``, then every ``n`` generations the algorithm will
            print the status of the generation. If verbose is set
            to -1, every generation information will be printed.

        labels : list of strings, default=[]
            (``ITExpr`` parameter) list containing the labels of the
            data that will be used in the evolutionary process, and
            will be used in ``ITExpr`` constructors.

        tfuncs : dict, default={'id': lambda x: x}
            (``ITExpr`` parameter) transformations functions to be
            used when creating ``ITExpr`` 's during the
            evolutionary process. Should always be a dict where the
            keys are the names of the transformation functions and 
            the values are unary vectorized functions (for example,
            numpy functions). For user-defined functions, see
            numpy.vectorize for more information on how to vectorize
            your transformation functions. Defaults to a dict with
            only the identity function.

        tfuncs_dx : dict, default=None
            (ITExpr_explainer parameter) derivatives of the
            given transformations functions, following the same scheme:
            a dictionary where the key is the name of the function
            (should have the derivatives of every function in
            tfuncs) and the value is a vectorized function
            representing its derivative. When set to None, the
            itea package will use automatic differentiation 
            through jax to create the derivatives.

        predictor_kw : dict or None, default = None
            dictionary with parameters to pass as named arguments
            to the constructor method in the  ``BaseITExpr`` subclass.
            If none is given, then a empty dict will be used.
        """

        self.gens            = gens
        self.popsize         = popsize
        self.max_terms       = max_terms
        self.expolim         = expolim
        self.tfuncs          = tfuncs
        self.tfuncs_dx       = tfuncs_dx
        self.random_state    = random_state
        self.labels          = labels
        self.verbose         = verbose
        self.simplify_method = simplify_method
        self.predictor_kw    = predictor_kw

        # This should always be none for the base class, so the default
        # fitness function for each task (regression/classification) is
        # correctly used.
        self.fitness_f = None


    def _check_args(self, X, y):
        """This method provides a simple verification of the arguments to be
        used as a baseline. 
        
        The sub-classes of the BaseITEA should implement the check_args as well.

        It is important to notice that the check must be made when fitting and
        should raise errors to stop the program flow if any problem is found.
        The scikit recomendation is to never do checks on __init__.

        Raises
        ------
            ValueError
                If one or more arguments would result in a invalid execution of
                itea.

        Notes
        -----
        As the scikit documentation suggests, no check for valid arguments is
        made on the constructor. Instead, when those arguments will be used, 
        we then perform the checkings. **All 'private' methods (beginning with
        an underscore) are designed to work after the ``check_args`` is called,
        since they rely on valid parameters. Also, all of them are intended
        to internal usage. When modifying, calling them directly, or testing
        the private methods, you should manually call the check args.
        """
        
        if self.expolim[1] < self.expolim[0]:
            raise ValueError(
                "Lower expolim bound is greater than upper bound.")

        if self.max_terms < 1:
            raise ValueError("max_terms should be greater or equal to 1.")

        for bound in self.expolim:
            if not np.issubdtype(type(bound), int):
                raise ValueError(
                    f"the expolim bounds {bound} must be integers.")

        if not np.issubdtype(type(self.max_terms), int):
            raise ValueError(f"max_terms should be a int.") 

        if self.simplify_method is not None:
            if not self.simplify_method in simplifiers.__all__:
                raise ValueError(
                    f"simplify_method {self.simplify_method} does not exist. "
                    f"Available methods: {simplifiers.__all__}")
        
        if 'id' not in list(self.tfuncs.keys()):
            warnings.warn("It is necessary to provide an identity function "
            "with name 'id' on the tfuncs dict, and I didn't found it. I will "
            "insert ``'id' : lambda x: x`` on the dict.")

            self.tfuncs['id'] = lambda x: x

        self.labels = np.array([self.labels]).flatten()

        if len(self.labels) != len(X[0]):
            warnings.warn("The labels vector does not have the same length as "
            "the number of variables in X (or was not provided). labels "
            f"has length {len(self.labels)}, and X has {len(X[0])} variables. "
            "labels will be generated as [x_0, x_1, ...].")
            
            self.labels = [f'x_{i}' for i in range(len(X[0]))]

        if self.predictor_kw == None:
            self.predictor_kw = {}


    def _create_population(
        self, *, simplify_f, nvars, itexpr_class, X, y, random_state):
        """Method to create an initial population for the evolutionary process.

        It will use an random expression generator that does not create
        trivial expressions (where all exponents are zero).

        Although, if the user has chosen an simplification method, exists the
        possibility that the initial population will have fewer individuals
        than the given popsize. The while loop tries to guarantee that we will
        start with a clean population where all fitnessess are finite values.
        """
        
        generator = uniform(
            self.max_terms, self.expolim, self.tfuncs, nvars, random_state)

        # The loop below ensures that the first population is always 
        # composed of valud expressions with finite fitness.
        pop = []
        while(len(pop) < self.popsize):
            expr = sanitize(next(generator))
            
            itexpr = itexpr_class(
                expr=sanitize(expr), tfuncs=self.tfuncs, labels=self.labels,
                fitness_f = self.fitness_f, **self.predictor_kw
            )
    
            with np.errstate(all='ignore'):
                itexpr.fit(X, y)

                if simplify_f is not None:
                    itexpr = simplify_f(itexpr=itexpr, X=X)
                    itexpr.fit(X, y)
            
            if np.isfinite(itexpr._fitness):
                pop.append(itexpr)
    
        return pop


    def _mutate_population(self, *, pop, nvars, itexpr_class, random_state):
        """Method to mutate the population without changing its parents.

        The mutated children will not be fitted. The fit of the ITExpr occurs
        only when the selection method faces an unfitted ITExpr.
        """

        mutated = [mutate_individual(p.expr, self.max_terms, self.expolim,
                          self.tfuncs, nvars, random_state) for p in pop]

        newpop = [itexpr_class(
                    expr = sanitize(expr), tfuncs = self.tfuncs,
                    labels = self.labels,
                    fitness_f = self.fitness_f, **self.predictor_kw
                    ) for expr in mutated]

        return newpop


    def _select_population(self, *,
        pop, select_f, simplify_f, size, X, y, random_state):
        """Method to perform multiple tournament selections, until the number
        of selected expressions is equal to the popsize.

        After selection, all individuals in the returned population are fitted,
        but not all individuals in the passed population will be as well.
        """
    
        # Invalid expressions can happen. We'll ignore the warnings just here
        with np.errstate(all='ignore'):

            # Selecting the competitors indexes. Even if there is only 1
            # individual in the population, the mutation will create a variation
            # and the population passed as argument (which is assumed to be
            # the concatenation of the original population and its mutated
            # children) should have at least 2 individuals to compete.
            competitors_idx = random_state.choice(len(pop), size=(size, 2))

            # Finding the unique expression in population that were selected
            # (to avoid unecessary fits)
            to_fit = np.unique(competitors_idx)

            # Simplify functions changes the expressions, we need to ensure
            # they will be fitted after the process
            if simplify_f is not None:
                for i in to_fit:
                    pop[i] = simplify_f(itexpr=pop[i].fit(X, y), X=X)

            for i in to_fit:
                pop[i].fit(X, y)
        
        return [select_f(comp) for comp in np.take(pop, competitors_idx)]
    

    def _evolve(self, X, y, itexpr_class, greater_is_better):
        """Evolution process on an ITExpr population.

        Should be used on sub-classes, inside the fit function, to evolve the
        population.
        """

        # Getting ready...
        nvars = X.shape[1]

        random_state = check_random_state(self.random_state)
        
        # Takes an array of competitors and returns the most valuable to the
        # task
        if greater_is_better:
            # making sure infinite fitness are negative
            select_f = lambda comp: comp[np.argmax(
                [c._fitness if np.isfinite(c._fitness) else -np.inf for c in comp])]
        else: # smaller is better. By default invalid itexprs have +inf fitness
            select_f = lambda comp: comp[np.argmin([c._fitness for c in comp])] 

        if self.simplify_method is not None:
            simplify_f = getattr(simplifiers, self.simplify_method)
        else:
            simplify_f = None

        groups  = ['fitness', 'n_terms', 'complexity']
        columns = ['min', 'mean', 'std', 'max']

        self.convergence_ = {
            group:{col:[] for col in columns} for group in groups} 

        # Starting the evolution now!        
        self.exectime_ = time.time()

        pop = self._create_population(
            simplify_f   = simplify_f, 
            nvars        = nvars,
            itexpr_class = itexpr_class,
            X = X, 
            y = y,
            random_state = random_state)

        if self.verbose:
            print("gen | smallest fitness | mean fitness | highest fitness | "
                  "remaining time")

            print("-"*76)
            
            # Estimation of remaining time using a circular list
            last_5_times = np.full(shape=(5), fill_value = np.nan, dtype=float)

        for g in range(self.gens):
            t = time.time()

            child = self._mutate_population(
                pop = pop,
                nvars = nvars,
                itexpr_class = itexpr_class,
                random_state = random_state)

            pop = self._select_population(
                pop        = pop + child,
                size       = self.popsize,
                select_f   = select_f,
                simplify_f = simplify_f, 
                X = X,
                y = y,
                random_state = random_state)
            
            # After selection, all individuals in population are fitted
            # and have finite fitness
            fitnesses    = [p._fitness     for p in pop]
            n_terms      = [p.n_terms      for p in pop]
            complexities = [p.complexity() for p in pop]

            for group, data in zip(groups, [fitnesses, n_terms, complexities]):
                data_masked = np.ma.masked_invalid(data)
                self.convergence_[group]['min' ].append(np.min(data_masked))
                self.convergence_[group]['max' ].append(np.max(data_masked))
                self.convergence_[group]['mean'].append(np.mean(data_masked))
                self.convergence_[group]['std' ].append(np.std(data_masked))
                
            if (self.verbose and g%self.verbose==0) or self.verbose==-1:

                # Estimating remaining time
                last_5_times[g%5] = time.time() - t
                
                remaining = int(np.ceil(
                    np.nanmean(last_5_times) * (self.gens - g - 1)))
                
                remaining_str = f"{remaining//60}min{remaining % 60}sec"

                print("{:3d} | {:16.6f} | {:12.6f} | {:15.6f} | {:12s}".format(
                    g, np.min(fitnesses), np.mean(fitnesses),
                    np.max(fitnesses), remaining_str
                ))
           
        self.exectime_ = time.time() - self.exectime_

        # At this point, all individuals in the population are fitted.
        if greater_is_better:
            # making sure infinite fitness are negative
            return pop[np.argmax(
                [p._fitness if np.isfinite(p._fitness) else -np.inf for p in pop])]
        else: # smaller is better. By default invalid itexprs have +inf fitness
            return pop[np.argmin([p._fitness for p in pop])]


    def _explain_bestsol(self, itexpr, X, y):
        """Estimating feature importantes using the partial effect of the
        final best solution.

        After the evolution process, this method should be called to create
        the feature_importances on the expression
        """

        explainer = ITExpr_explainer(
            itexpr=itexpr, tfuncs=self.tfuncs, tfuncs_dx=self.tfuncs_dx
        ).fit(X, y)

        itexpr.selected_features_ = explainer.selected_features()

        itexpr.feature_importances_ = explainer.average_partial_effects(X)

        self.feature_importances_ = itexpr.feature_importances_


    def fit(self, X, y):
        """virtual fit method. Should be overridden by sub-classes.
        """

        # The subclasses must do:
        # 1 - check_args
        # 2 - run the evolution (with _evolve()) 
        # 3 - retrieve the best solution
        # 4 - calculate the feature importances of the best solution

        raise NotImplementedError()


    def predict(self, X):
        """virtual predict method. Should be overridden by sub-classes.
        """
        
        raise NotImplementedError()


    def predict_proba(self, X):
        """virtual predict_proba method. Should be overridden by sub-classes.
        """

        raise NotImplementedError()
