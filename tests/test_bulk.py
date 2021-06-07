"""Tests for :mod:`nanoCAT.recipes.bulk`."""

from __future__ import annotations

from typing import Mapping, Any

import numpy as np
import pytest
from assertionlib import assertion
from nanoCAT.recipes import bulk_workflow, fast_bulk_workflow

FAST_BULK_REF = np.array([
    [40.16235, 207.024513, 987.120114],
    [41.16235, 208.024513, 988.120114],
    [8.852858, 21.181752, 38.191051],
    [0.0, 0.0, 91.447878],
    [0.0, 0.0, 0.54337059],
    [0.0, 0.0, 749.750132],
    [0.0, 0.0, 749.750132],
    [0.0, 0.0, 749.750132],
], dtype=np.float64)
FAST_BULK_REF.setflags(write=False)


def test_bulk_workflow() -> None:
    """Tests for :func:`bulk_workflow`."""
    smiles_list = ['CO', 'CCO', 'CCCO']
    mol_list, bulk_ar = bulk_workflow(smiles_list, anchor='O[H]', diameter=1)

    formula_list = ['C1H3O1', 'C2H5O1', 'C3H7O1']
    iterator = enumerate(zip(mol_list, formula_list), start=2)
    for i, (mol, formula) in iterator:
        assertion.eq(mol[i].coords, (0.0, 0.0, 0.0))
        assertion.eq(mol.get_formula(), formula)

    ref = [17.07131616, 64.15117841, 74.79488029]
    np.testing.assert_allclose(bulk_ar, ref)


@pytest.mark.parametrize("kwargs,i", [
    (dict(diameter=1), 0),
    (dict(diameter=None), 1),
    (dict(func=None), 2),
    (dict(func=lambda x: x**2), 3),
    (dict(func=lambda x: 1/x), 4),
    (dict(height_lim=5), 5),
    (dict(height_lim=None), 6),
    ({}, 7),
])
def test_fast_bulk_workflow(kwargs: Mapping[str, Any], i: int) -> None:
    """Tests for :func:`fast_bulk_workflow`."""
    smiles_list = ['CO', 'CCO', 'CCCO']
    mol_list, bulk_ar = fast_bulk_workflow(smiles_list, anchor='O[H]', **kwargs)

    formula_list = ['C1H3O1', 'C2H5O1', 'C3H7O1']
    for mol, formula in zip(mol_list, formula_list):
        assertion.eq(mol.get_formula(), formula)

    ref = FAST_BULK_REF[i]
    np.testing.assert_allclose(bulk_ar, ref, rtol=1e-07)
