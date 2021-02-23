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

    @pytest.mark.parametrize(
        "name,kwargs",
        [
            (None, {}),
            ("count_y", {"count_y": 2}),
            ("count_x", {"count_x": 2}),
            ("n_pairs", {"n_pairs": 3}),
            ("mode", {"mode": "cluster"}),
        ],
        ids=["None", "count_y", "count_x", "n_pairs", "mode"],
    )
    def test_passes(self, name: None | str, kwargs: Mapping[str, Any]) -> None:
        mol = dissociate_bulk(MOL_PbBr, "Au", "Cl", k=6, **kwargs)

        if name is None:
            filename = "test_dissociate_bulk.xyz"
        else:
            filename = f"test_dissociate_bulk_{name}.xyz"
        ref = Molecule(PATH / filename)

        np.testing.assert_allclose(mol, ref, atol=0.00001)

        au_count = len([i for i in mol if i.symbol == "Au"])
        au_count_ref = len([i for i in MOL_PbBr if i.symbol == "Au"])
        au_count_ref -= kwargs.get("count_x", 1) * kwargs.get("n_pairs", 1)
        assertion.eq(au_count, au_count_ref)

        cl_count = len([i for i in mol if i.symbol == "Cl"])
        cl_count_ref = len([i for i in MOL_PbBr if i.symbol == "Cl"])
        cl_count_ref -= kwargs.get("count_y", 1) * kwargs.get("n_pairs", 1)
        assertion.eq(cl_count, cl_count_ref)

    @pytest.mark.parametrize(
        "kwargs,exc",
        [
            ({"symbol_x": "Au", "symbol_y": "Cl", "count_x": 0}, ValueError),
            ({"symbol_x": "Au", "symbol_y": "Cl", "count_x": 999}, ValueError),
            ({"symbol_x": "Au", "symbol_y": "Cl", "count_y": 0}, ValueError),
            ({"symbol_x": "H", "symbol_y": "Cl"}, MoleculeError),
            ({"symbol_x": "Au", "symbol_y": "H"}, MoleculeError),
            ({"symbol_x": "Au", "symbol_y": "Cl", "count_x": 2, "cluster_size": 2}, TypeError),
        ],
        ids=["count_x_0", "count_x_999", "count_y", "symbol_x", "symbol_y", "cluster_size"],
    )
    def test_raises(self, kwargs: Mapping[str, Any], exc: Type[Exception]) -> None:
        with pytest.raises(exc):
            dissociate_bulk(MOL_PbBr, **kwargs)
