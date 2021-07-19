import pytest

import numpy as np

from sklearn.base import clone

from itea._base import BaseITEA
from itea.regression import ITExpr_regressor


def test_BaseITEA_check_args():
    # Testing only for raises. The check_args also generate warnings,
    # but the algorithm can execute normally if no exception was thrown. 

    X = np.array([[1, 1, 1]])
    y = np.array([3])

    with pytest.raises(ValueError) as execinfo:
        baseitea = BaseITEA(expolim=(2, 1))._check_args(X, y)

    assert execinfo.value.args[0] == ("Lower expolim bound is greater "
                                      "than upper bound.")

    for expolim in [(1, 1.5), (2.0, 3)]:
        with pytest.raises(ValueError) as execinfo:
            baseitea = BaseITEA(
                expolim=expolim, max_terms=2)._check_args(X, y)

        assert "must be integers" in execinfo.value.args[0]

    with pytest.raises(ValueError) as execinfo:
        baseitea = BaseITEA(
            expolim=(1, 2), max_terms=0)._check_args(X, y)

    assert "max_terms should be greater or equal" in execinfo.value.args[0]
    
    with pytest.raises(ValueError) as execinfo:
        baseitea = BaseITEA(
            expolim=(1, 1), max_terms=2.0)._check_args(X, y)

    assert "max_terms should be a int." == execinfo.value.args[0]

    with pytest.raises(ValueError) as execinfo:
        baseitea = BaseITEA(simplify_method='I_do_not_exist')._check_args(X, y)

    assert "I_do_not_exist does not exist" in execinfo.value.args[0]


# NOTE: all private methods (beginning with _) takes as argument a random
# state, and does not handle different arguments. Only the _evolve method
# (which calls all the others) checks if the random_state is a variable
# or a numpy.randomState and makes the appropriate management to call the
# others. The following tests will use a randomstate instance, instead of
# the usual number. 
# Before calling any method that begins with an underscore, it is important
# to call _check_args, to do the proper adjustments before execution.

def test_population_size():
    X, y = np.array([[1.0, 2.0]]), np.array([3.0])

    baseitea = BaseITEA(popsize=10, labels=['x0', 'x1'])
    baseitea._check_args(X, y)

    # we need to provide a subclass of BaseITExpr. Using the regressor just
    # to test the BaseITEA, since it is faster than the classification

    pop = baseitea._create_population(nvars=2, simplify_f=None, X=X, y=y,
        itexpr_class=ITExpr_regressor, random_state=np.random.RandomState(42))

    assert len(pop) == baseitea.get_params()['popsize']


def test_mutation_size():
    X, y = np.array([[1.0, 2.0]]), np.array([3.0])

    baseitea = BaseITEA(popsize=10, labels=['x0', 'x1'])
    baseitea._check_args(X, y)

    pop = baseitea._create_population(nvars=2, simplify_f=None, X=X, y=y,
        itexpr_class=ITExpr_regressor, random_state=np.random.RandomState(42))

    # individual mutations are tested in test_mutations. This test assumes
    # the mutations works as they should.
    xmen = baseitea._mutate_population(pop=pop, nvars=2,
        itexpr_class=ITExpr_regressor, random_state=np.random.RandomState(42))

    assert len(xmen) == baseitea.get_params()['popsize']


def test_mutation_dont_change_originals():
    X, y = np.array([[1.0, 2.0]]), np.array([3.0])

    baseitea = BaseITEA(popsize=50, labels=['x0', 'x1'])
    baseitea._check_args(X, y)

    pop = baseitea._create_population(nvars=2, simplify_f=None, X=X, y=y,
        itexpr_class=ITExpr_regressor, random_state=np.random.RandomState(42))

    # clone is tested on test_BaseITExpr
    pop_backup = [clone(p) for p in pop]

    xmen = baseitea._mutate_population(pop=pop, nvars=2,
        itexpr_class=ITExpr_regressor, random_state=np.random.RandomState(42))

    for p1, p2 in zip(pop, pop_backup):
        for (f1, t1), (f2, t2) in zip(p1.expr, p2.expr):
            assert f1 == f2
            assert t1 == t2


def test_selection():
    X, y = np.array([[1.0, 2.0]]), np.array([3.0])

    baseitea = BaseITEA(popsize=10, labels=['x0', 'x1'])
    baseitea._check_args(X, y)

    pop = baseitea._create_population(nvars=2, simplify_f=None, X=X, y=y,
        itexpr_class=ITExpr_regressor, random_state=np.random.RandomState(42))
    
    select_f = lambda comp: comp[np.argmin([c._fitness for c in comp])] 

    selected = baseitea._select_population(pop=pop, select_f=select_f,
        simplify_f=None, size=baseitea.get_params()['popsize'],
        X=X, y=y, random_state=np.random.RandomState(42))

    for s in selected:
        assert s in pop

        # After the selection, every individual should be fitted.
        assert s._is_fitted


def test_random_state():
    # testing the whole evolution random state also guarantees that mutation
    # and selection random states will work as they should.

    X, y = np.array([[1.0, 2.0]]), np.array([3.0])

    baseitea_1 = BaseITEA(popsize=10, gens=10,
        labels=['x0', 'x1'], random_state=42)

    baseitea_1._check_args(X, y)

    bestsol_1 = baseitea_1._evolve(
        X=X, y=y, itexpr_class=ITExpr_regressor, greater_is_better=False)

    baseitea_2 = BaseITEA(popsize=10, gens=10,
        labels=['x0', 'x1'], random_state=42)    

    baseitea_2._check_args(X, y)

    bestsol_2 = baseitea_2._evolve(
        X=X, y=y, itexpr_class=ITExpr_regressor, greater_is_better=False)

    assert str(bestsol_1) == str(bestsol_2)


def test_virtual_methods():
    X, y = np.array([[1.0, 2.0]]), np.array([3.0])

    baseitea = BaseITEA(labels=['x0', 'x1'])
    baseitea._check_args(X, y)

    with pytest.raises(NotImplementedError):
        baseitea.fit(X, y)

    with pytest.raises(NotImplementedError):
        baseitea.predict(X)

    with pytest.raises(NotImplementedError):
        baseitea.predict_proba(X)