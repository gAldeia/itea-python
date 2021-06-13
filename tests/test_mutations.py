import pytest

import numpy as np

from itea._manipulators.generator import uniform
from itea._manipulators.mutation  import (_mutation_add, _mutation_drop,
    _mutation_intern, _mutation_interp, _mutation_term)


# Setting a simple configuration to evaluate the methods
max_terms = 5
expolim  = (-2, 2)
tfuncs   = {'id': lambda x: x, 'sin': np.sin}
nvars    = 5


def test_mutation_drop_valid_exprs():

    generator  = uniform(
        max_terms, expolim, tfuncs, nvars, np.random.RandomState(42))
    
    population = [p for p in [next(generator) for _ in range(50)] if len(p)> 1]
    
    xmen = [
        _mutation_drop(p, expolim, tfuncs, nvars, np.random.RandomState(15))
        for p in population
    ]

    for m, p in zip(xmen, population):
        assert type(m) == list

        assert len(m) == len(p) - 1
        assert len(m) > 0

        for f, t in m:
            assert f in tfuncs.keys()
            assert all(ti <= expolim[1] for ti in t)
            assert all(ti >= expolim[0] for ti in t)


def test_mutation_term_valid_exprs():
    generator  = uniform(
        max_terms, expolim, tfuncs, nvars, np.random.RandomState(42))
    
    population = [next(generator) for _ in range(50)]
    
    xmen = [
        _mutation_term(p, expolim, tfuncs, nvars, np.random.RandomState(15))
        for p in population
    ]

    for m, p in zip(xmen, population):
        assert type(m) == list

        assert len(m) == len(p)
        assert len(m) > 0

        for f, t in m:
            assert f in tfuncs.keys()
            assert all(ti <= expolim[1] for ti in t)
            assert all(ti >= expolim[0] for ti in t)

        # Only one term is changed, and this occurs at the same position as
        # the term being changed on the original expr
        for m, p in zip(xmen, population):
            equal = 0
            for (f1, t1), (f2, t2) in zip(m, p):
                if f1 == f2 and t1 == t2:
                    equal += 1

            # greater or equal, because the mutation can end up creating an
            # identical new term
            assert equal >= len(m) - 1


@pytest.mark.parametrize(
    'mutation_f', [_mutation_add, _mutation_interp, _mutation_intern])
def test_incremental_mutation(mutation_f):
    generator  = uniform(
        max_terms, expolim, tfuncs, nvars, np.random.RandomState(42))
    
    population = [next(generator) for _ in range(50)]

    xmen = [
        mutation_f(p, expolim, tfuncs, nvars, np.random.RandomState(15))
        for p in population
    ]

    for m, p in zip(xmen, population):
        assert type(m) == list

        assert len(m) == len(p) + 1
        assert len(m) > 0

        for f, t in m:
            assert f in tfuncs.keys()
            assert all(ti <= expolim[1] for ti in t)
            assert all(ti >= expolim[0] for ti in t)

        # the new term is always added to the end of the expr list. zipping
        # the original and mutated should end up in equal expressions
        for m, p in zip(xmen, population):
            for (f1, t1), (f2, t2) in zip(m, p):
                assert f1 == f2
                assert t1 == t2


@pytest.mark.parametrize(
    'mutation_method', [_mutation_add, _mutation_drop, _mutation_intern,
    _mutation_interp, _mutation_term])
def test_mutations_dont_change_originals(mutation_method):

    generator  = uniform(
        max_terms, expolim, tfuncs, nvars, np.random.RandomState(42))
    
    population = [p for p in [next(generator) for _ in range(50)] if len(p)> 1]
    
    population_backup = [p.copy() for p in population]
    
    xmen = [
        mutation_method(p, expolim, tfuncs, nvars, np.random.RandomState(15))
        for p in population
    ]

    # original exprs shoudnt be affected by mutation
    for p1, p2 in zip(population, population_backup):
        for (f1, t1), (f2, t2) in zip(p1, p2):
            assert f1 == f2
            assert t1 == t2


@pytest.mark.parametrize(
    'mutation_method', [_mutation_add, _mutation_drop, _mutation_intern,
    _mutation_interp, _mutation_term])
def test_mutations_reproductible_random_state(mutation_method):

    generator  = uniform(
        max_terms, expolim, tfuncs, nvars, np.random.RandomState(42))
    
    population = [p for p in [next(generator) for _ in range(50)] if len(p)> 1]
    
    population_backup = [p.copy() for p in population]
    
    xmen = [
        mutation_method(p, expolim, tfuncs, nvars, np.random.RandomState(15))
        for p in population
    ]

    xmen2 = [
        mutation_method(p, expolim, tfuncs, nvars, np.random.RandomState(15))
        for p in population
    ]

    for m1, m2 in zip(xmen, xmen2):
        for (f1, t1), (f2, t2) in zip(m1, m2):
            assert f1 == f2
            assert t1 == t2