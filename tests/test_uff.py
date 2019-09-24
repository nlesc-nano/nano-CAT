"""Tests for :mod:`nanoCAT.ff.uff`."""

from os.path import join

import numpy as np
import pandas as pd

from CAT.assertion.assertion_manager import assertion
from nanoCAT.ff.uff import combine_xi, combine_di, UFF_DF

PATH: str = join('tests', 'test_files')


def test_combine_xi() -> None:
    """Tests for :func:`nanoCAT.ff.uff.combine_xi`."""
    xi_ref = np.load(join(PATH, 'xi.npy'))
    xi = np.array([combine_xi('H', i) for i in UFF_DF.index])

    np.testing.assert_allclose(xi_ref, xi)
    assertion.exception(KeyError, combine_xi, 'bob', 'bob')


def test_combine_di() -> None:
    """Tests for :func:`nanoCAT.ff.uff.combine_di`."""
    di_ref = np.load(join(PATH, 'di.npy'))
    di = np.array([combine_di('H', i) for i in UFF_DF.index])

    np.testing.assert_allclose(di_ref, di)
    assertion.exception(KeyError, combine_di, 'bob', 'bob')


def test_uff_df() -> None:
    """Tests for :data:`nanoCAT.ff.uff.UFF_DF`."""
    assertion.isinstance(UFF_DF, pd.DataFrame)
    assertion.eq(UFF_DF.shape, (103, 3))
    for _, v in UFF_DF.items():
        assertion.eq(v.dtype, np.dtype(float))
