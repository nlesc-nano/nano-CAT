"""Tests for :mod:`nanoCAT.recipes.bulk`."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any

import numpy as np
import pandas as pd
import pytest
from assertionlib import assertion
from nanoCAT.recipes import bulk_workflow, fast_bulk_workflow

PATH = Path("tests") / "test_files"
FAST_BULK_REF = pd.read_csv(PATH / "bulk.csv", index_col=0)
FAST_BULK_REF.columns = FAST_BULK_REF.columns.astype(np.int64)


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


@pytest.mark.parametrize("kwargs,i,j", [
    (dict(diameter=1), 0, 15),
    (dict(diameter=None), 1, 15),
    (dict(func=None), 2, 15),
    (dict(func=lambda x: x**2), 3, 15),
    (dict(func=lambda x: 1/x), 4, 15),
    (dict(height_lim=5), 5, 15),
    (dict(height_lim=None), 6, 15),
    ({}, 7, None),
])
def test_fast_bulk_workflow(kwargs: Mapping[str, Any], i: int, j: None | int) -> None:
    """Tests for :func:`fast_bulk_workflow`."""
    smiles_list = FAST_BULK_REF.index[:j]
    _, bulk_ar = fast_bulk_workflow(smiles_list, **kwargs)

    ref = FAST_BULK_REF.iloc[:j, i]
    np.testing.assert_allclose(bulk_ar, ref)
