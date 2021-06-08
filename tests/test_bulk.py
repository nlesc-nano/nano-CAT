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

SMILES = pd.read_csv(PATH / "bulk_input.csv")["smiles"]
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


@pytest.mark.xfail(reason="Needs more work", raises=AssertionError)
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
    _, bulk_ar = fast_bulk_workflow(SMILES, **kwargs)
    ref = FAST_BULK_REF[i]
    np.testing.assert_allclose(bulk_ar, ref)
