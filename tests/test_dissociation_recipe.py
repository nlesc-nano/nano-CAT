"""Tests for :mod:`nanoCAT.recipes.dissociate_surface`."""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Mapping, Any, Type

import pytest
import numpy as np

from scm.plams import Molecule, PTError, MoleculeError
from assertionlib import assertion
from nanoCAT.recipes import dissociate_surface, dissociate_bulk

PATH = Path('tests') / 'test_files'
MOL = Molecule(PATH / 'Cd360Se309.xyz')
XYZ = np.array(MOL)

MOL_PbBr = Molecule(PATH / '0D+PbBr2.xyz')
for at in MOL_PbBr:
    if at.symbol in {"I", "Br"}:
        at.symbol = "Cl"


def test_dissociate_surface() -> None:
    """Tests for :func:`dissociate_surface`."""
    idx_tup = (
        319,

        [319],

        [320, 319],

        [[320, 319],
         [158, 57],
         [156, 155]]
    )

    at_idx_iter = iter([
        319,
        319,
        320, 319,
        320, 319, 158, 57, 156, 155
    ])

    mol_iter = chain.from_iterable(dissociate_surface(MOL, i) for i in idx_tup)
    for i, mol in zip(at_idx_iter, mol_iter):
        assertion.contains(np.asarray(mol), XYZ[i], invert=True)

    assertion.assert_(next, dissociate_surface(MOL, i, k=0), exception=ValueError)
    assertion.assert_(next, dissociate_surface(MOL, i, k=999), exception=ValueError)
    assertion.assert_(next, dissociate_surface(MOL, i, lig_count=999), exception=ValueError)
    assertion.assert_(next, dissociate_surface(MOL, i, lig_count=-1), exception=ValueError)
    assertion.assert_(next, dissociate_surface(MOL, i, symbol='bob'), exception=PTError)
    assertion.assert_(next, dissociate_surface(MOL, i, symbol=999), exception=PTError)
    assertion.assert_(next, dissociate_surface(MOL, i, symbol=9.5), exception=TypeError)


class TestDissociateBulk:
    """Tests for :func:`dissociate_bulk`."""

    @pytest.mark.parametrize("count_x", [1, 2])
    @pytest.mark.parametrize("count_y", [1, 2])
    @pytest.mark.parametrize("k", [None, 6])
    @pytest.mark.parametrize("r_max", [None, 10])
    @pytest.mark.parametrize("n_pairs", [0, 1, 2])
    def test_passes(
        self,
        count_x: int,
        count_y: int,
        k: None | int,
        r_max: None | float,
        n_pairs: int,
    ) -> None:
        if k is None and r_max is None:
            return None

        mol = dissociate_bulk(
            MOL_PbBr, "Au", "Cl", k=k, r_max=r_max,
            count_x=count_x, count_y=count_y, n_pairs=n_pairs,
        )

        filename = f"test_dissociate_bulk_{count_x}{count_y}{n_pairs}{k}{r_max}.xyz"
        ref = Molecule(PATH / 'dissociate' / filename)

        np.testing.assert_allclose(mol, ref, atol=0.00001)

        au_count = len([i for i in mol if i.symbol == "Au"])
        au_count_ref = len([i for i in MOL_PbBr if i.symbol == "Au"])
        au_count_ref -= count_x * n_pairs
        assertion.eq(au_count, au_count_ref)

        cl_count = len([i for i in mol if i.symbol == "Cl"])
        cl_count_ref = len([i for i in MOL_PbBr if i.symbol == "Cl"])
        cl_count_ref -= count_y * n_pairs
        assertion.eq(cl_count, cl_count_ref)
        return None

    @pytest.mark.parametrize(
        "kwargs,exc",
        [
            ({"symbol_x": "Au", "symbol_y": "Cl", "count_x": 0}, ValueError),
            ({"symbol_x": "Au", "symbol_y": "Cl", "count_x": 999}, ValueError),
            ({"symbol_x": "Au", "symbol_y": "Cl", "count_y": -1}, ValueError),
            ({"symbol_x": "Au", "symbol_y": "Cl", "count_y": 999}, ValueError),
            ({"symbol_x": "Au", "symbol_y": "Cl", "n_pairs": -1}, ValueError),
            ({"symbol_x": "Au", "symbol_y": "Cl", "n_pairs": 999}, ValueError),
            ({"symbol_x": "H", "symbol_y": "Cl"}, MoleculeError),
            ({"symbol_x": "Au", "symbol_y": "H"}, MoleculeError),
            ({"symbol_x": "Au", "symbol_y": "Cl", "count_x": 2, "cluster_size": 2}, TypeError),
            ({"symbol_x": "Au", "symbol_y": "Cl", "k": None, "r_max": None}, TypeError),
            ({"symbol_x": "Au", "symbol_y": "Cl", "count_y": 5, "r_max": 1.0}, ValueError),
        ],
    )
    def test_raises(self, kwargs: Mapping[str, Any], exc: Type[Exception]) -> None:
        with pytest.raises(exc):
            dissociate_bulk(MOL_PbBr, **kwargs)
