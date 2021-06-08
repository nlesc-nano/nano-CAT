"""Tests for :mod:`nanoCAT.recipes.bulk`."""

from __future__ import annotations

import sys
from pprint import pformat
from pathlib import Path
from typing import Tuple, List, Mapping, TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from assertionlib import assertion
from scm.plams import Molecule
from nanoCAT.recipes import bulk_workflow, fast_bulk_workflow

if TYPE_CHECKING:
    import _pytest
    import numpy.typing as npt

    Output = Tuple[
        List[Molecule],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]

WIN_OR_LINUX = sys.platform in {"win32", "linux"}

PATH = Path("tests") / "test_files"

SMILES = pd.read_csv(PATH / "bulk_input.csv")["smiles"]
FAST_BULK_REF = pd.read_csv(PATH / "bulk.csv", index_col=0)
FAST_BULK_REF.columns = FAST_BULK_REF.columns.astype(np.int64)

MOL_ARRAY_REF: npt.NDArray[np.float64] = np.load(PATH / "bulk_array.npy")
BOND_ARRAY_REF: npt.NDArray[np.float64] = np.load(PATH / "bond_array.npy")
SYMBOL_ARRAY_REF: npt.NDArray[np.str_] = np.load(PATH / "symbol_array.npy")


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


def eval_ex_dict(ex_dict: Mapping[str, AssertionError], i: int) -> None:
    """Evaluate the content of **ex_dict** and raise if it contains any exceptions."""
    __traceback_hide__ = True  # noqa: F841
    if not len(ex_dict):
        return None

    ex = next(iter(ex_dict.values()))
    failed = pformat(list(ex_dict.keys()))
    msg = f"Failed for {len(ex_dict)}/{i} molecules\n{failed}"
    raise AssertionError(msg) from ex


class TestFastBulkWorkflow:
    """Tests for :func:`fast_bulk_workflow`."""

    IDS_MAPPING = {
        "diameter=1": dict(diameter=1),
        "diameter=None": dict(diameter=None),
        "func=None": dict(func=None),
        "func=lambda x: x**2": dict(func=lambda x: x**2),
        "func=lambda x: 1/x": dict(func=lambda x: 1/x),
        "height_lim=5": dict(height_lim=5),
        "height_lim=None": dict(height_lim=None),
        "": {},
    }
    IDS = IDS_MAPPING.keys()
    PARAMS = enumerate(IDS_MAPPING.values())

    @pytest.fixture(scope="class", autouse=True, ids=IDS, params=PARAMS)
    def generate_output(self, request: _pytest.fixtures.SubRequest) -> Output:
        """Generate the test output of :func:`fast_bulk_workflow`."""
        i, kwargs = request.param
        mol_list, v_bulk = fast_bulk_workflow(SMILES, **kwargs)
        return mol_list, v_bulk, FAST_BULK_REF[i].values

    def test_mol_properties(self, generate_output: Output) -> None:
        """Validate the properties of the output molecules."""
        mol_list, *_ = generate_output
        ex_dict = {}
        for i, mol in enumerate(mol_list):
            smiles: None | str = mol.properties.get("smiles")
            anchor: None | str = mol.properties.get("anchor")
            name = smiles if smiles is not None else mol.get_formula()
            try:
                assertion.isinstance(smiles, str, message=f"{name} smiles")
                assertion.isinstance(anchor, str, message=f"{name} anchor")
            except AssertionError as ex:
                ex_dict[name] = ex
        eval_ex_dict(ex_dict, 1 + i)

    @pytest.mark.xfail(WIN_OR_LINUX, reason="Windows or linux", raises=AssertionError, strict=True)
    def test_mol_coords(self, generate_output: Output) -> None:
        """Validate the Cartesian coordinates of the output molecules."""
        mol_list, *_ = generate_output
        ex_dict = {}
        for i, mol in enumerate(mol_list):
            name = mol.properties.smiles
            try:
                np.testing.assert_allclose(
                    actual=mol,
                    desired=MOL_ARRAY_REF[i, :len(mol)],
                    err_msg=name + "\n",
                )
            except AssertionError as ex:
                ex_dict[name] = ex
        eval_ex_dict(ex_dict, 1 + i)

    def test_mol_bonds(self, generate_output: Output) -> None:
        """Validate the bonds and bond orders of the output molecules."""
        mol_list, *_ = generate_output
        ex_dict = {}
        for i, mol in enumerate(mol_list):
            name = mol.properties.smiles
            try:
                np.testing.assert_allclose(
                    actual=mol.bond_matrix(),
                    desired=BOND_ARRAY_REF[i, :len(mol), :len(mol)],
                    err_msg=mol.properties.smiles + "\n",
                )
            except AssertionError as ex:
                ex_dict[name] = ex
        eval_ex_dict(ex_dict, 1 + i)

    def test_mol_symbols(self, generate_output: Output) -> None:
        """Validate the atomic symbols of the output molecules."""
        mol_list, *_ = generate_output
        ex_dict = {}
        for i, mol in enumerate(mol_list):
            name = mol.properties.smiles
            try:
                np.testing.assert_array_equal(
                    x=[at.symbol for at in mol],
                    y=SYMBOL_ARRAY_REF[i, :len(mol)],
                    err_msg=mol.properties.smiles + "\n",
                )
            except AssertionError as ex:
                ex_dict[name] = ex
        eval_ex_dict(ex_dict, 1 + i)

    @pytest.mark.xfail(raises=AssertionError)
    def test_bulk(self, generate_output: Output) -> None:
        """Validate the output bulkiness values."""
        _, bulk_ar, ref = generate_output
        np.testing.assert_allclose(bulk_ar, ref)
