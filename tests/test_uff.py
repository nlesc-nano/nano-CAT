"""Tests for :mod:`nanoCAT.ff.uff`."""

from os.path import join

import numpy as np
import pandas as pd

from CAT.assertion_functions import assert_instance, assert_eq, assert_exception
from nanoCAT.ff.uff import combine_xi, combine_di, UFF_DF

PATH: str = join('tests', 'test_files')


def test_combine_xi() -> None:
    """Tests for :func:`nanoCAT.ff.uff.combine_xi`."""
    xi_ref = np.load(join(PATH, 'xi.npy'))
    xi = np.array([combine_xi('H', i) for i in UFF_DF.index])
    np.testing.assert_allclose(xi_ref, xi)
    assert_exception(KeyError, combine_xi, 'bob', 'bob')


def test_combine_di() -> None:
    """Tests for :func:`nanoCAT.ff.uff.combine_di`."""
    di_ref = np.load(join(PATH, 'di.npy'))
    di = np.array([combine_di('H', i) for i in UFF_DF.index])
    np.testing.assert_allclose(di_ref, di)
    assert_exception(KeyError, combine_di, 'bob', 'bob')


def test_uff_df() -> None:
    """Tests for :data:`nanoCAT.ff.uff.UFF_DF`."""
    assert_instance(UFF_DF, pd.DataFrame)
    assert_eq(UFF_DF.shape, (103, 3))
    for _, v in UFF_DF.items():
        assert_eq(v.dtype, np.dtype(float))
