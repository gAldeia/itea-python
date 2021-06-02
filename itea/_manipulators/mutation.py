# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 05-30-2021 by Guilherme Aldeia


"""Mutation methods.

The mutations are the core evolutionary aspect of ITEA, since the algorithm
does not have an crossover operator. This file implements several mutations
that defines different ways of exploring the neighborhood of the expressions.

All mutation methods should work with an expr (list os IT terms, 
not an instance of ``ITExpr``), and does not change the expr list passed as 
argument. Instead, it should create a new list to avoid collateral effects.

For more informations, please see the original itea paper:
"França, F., & Aldeia, G. (2020).
Interaction-Transformation Evolutionary Algorithm for Symbolic Regression.
Evolutionary Computation, 1-25."
"""


from itea._manipulators.generator import uniform


__all__ = [
    'mutate_individual'
]


def _mutation_drop(expr, expolim, tfuncs, nvars, random_state, **kwargs):
    """Randomly removes one IT term from the IT expression"""

    index = random_state.randint(0, len(expr))

    return [expr[i] for i in range(len(expr)) if i!= index]


def _mutation_add(expr, expolim, tfuncs, nvars, random_state, **kwargs):
    """Generates a new term and inserts it at the end of the IT expression"""

    randITGenerator = uniform(1, expolim, tfuncs, nvars, random_state)

    newterm = next(randITGenerator)[0]
    
    return expr + [newterm]


def _mutation_term(expr, expolim, tfuncs, nvars, random_state, **kwargs):
    """Randomly replaces an exponent from the IT expression.
    
    If the new exponent transforms the term into a trivial one (with all
    exponents equals to zero), then a new random term is created. The generator
    ensures that no trivial term will be created.
    """

    indexTerm     = random_state.randint(0, len(expr))
    indexStrength = random_state.randint(0, nvars)
    newStrength   = random_state.randint(expolim[0], expolim[1] + 1)

    oldf, oldt = expr[indexTerm]
    
    newt = oldt.copy()
    newt[indexStrength] = newStrength
    
    if all(t == 0 for t in newt):    
        replacer = uniform(1, expolim, tfuncs, nvars, random_state)
        newf, newt = next(replacer)[0]

    return [expr[i] if i!= indexTerm else (oldf, newt)
            for i in range(len(expr))]


def _mutation_interp(expr, expolim, tfuncs, nvars, random_state, **kwargs):
    """Randomly select two terms to be 'merged': their exponents arrays
    will be added element-wise, and the transformation function of the
    first randomly selected term will be used.

    If the new exponent transforms the term into a trivial one (with all
    exponents equals to zero), then a new random term is created. The generator
    ensures that no trivial term will be created.
    """

    fst_index, snd_index = random_state.randint(0, len(expr), size=2)

    fstf, fstt = expr[fst_index]
    sndt, sndt = expr[snd_index]

    # Avoiding values outside the expolim boundries
    newt = [min(max(fstt[i] + sndt[i], expolim[0]), expolim[1])
            for i in range(nvars)]
    
    if all(t == 0 for t in newt):    
        replacer = uniform(1, expolim, tfuncs, nvars, random_state)
        newf, newt = next(replacer)[0]

    return expr + [(fstf, newt)]


def _mutation_intern(expr, expolim, tfuncs, nvars, random_state, **kwargs):
    """Randomly select two terms to be 'merged': their exponents arrays
    will be subtracted element-wise, and the transformation function of the
    first randomly selected term will be used.

    If the new exponent transforms the term into a trivial one (with all
    exponents equals to zero), then a new random term is created. The generator
    ensures that no trivial term will be created.
    """

    fst_index, snd_index = random_state.randint(0, len(expr), size=2)

    fstf, fstt = expr[fst_index]
    sndf, sndt = expr[snd_index]

    newt = [min(max(fstt[i] - sndt[i], expolim[0]), expolim[1])
            for i in range(nvars)]
    
    if all(t == 0 for t in newt):    
        replacer = uniform(1, expolim, tfuncs, nvars, random_state)
        newf, newt = next(replacer)[0]

    return expr + [(fstf, newt)]


def mutate_individual(
    expr, max_terms, expolim, tfuncs, nvars, random_state, **kwargs):
    """Mutation function that takes an expr and, based on the length, selects
    one random mutation to be applied on the expr.

    This is done to respect the specifier max_number of terms.
    """
    
    applicable = [_mutation_term]

    if len(expr) > 1:
        applicable += [_mutation_drop]
    if len(expr) < max_terms:
        applicable += [_mutation_add, _mutation_intern, _mutation_interp]

    return random_state.choice(applicable)(
        expr, expolim, tfuncs, nvars, random_state, **kwargs
    )