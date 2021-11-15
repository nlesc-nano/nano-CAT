"""Tests for :mod:`nanoCAT.recipes.bulk`."""

from __future__ import annotations

from pprint import pformat
from pathlib import Path
from typing import Mapping, TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
import pytest
from assertionlib import assertion
from scm.plams import Molecule
from scipy.spatial.distance import cdist
from nanoCAT.recipes import bulk_workflow, fast_bulk_workflow

if TYPE_CHECKING:
    import _pytest
    from numpy.typing import NDArray
    from numpy import void as V

PATH = Path("tests") / "test_files"

SMILES = pd.read_csv(PATH / "bulk_input.csv")["smiles"]
FAST_BULK_REF = pd.read_csv(PATH / "bulk.csv", index_col=0)
FAST_BULK_REF.columns = FAST_BULK_REF.columns.astype(np.int64)

MOL_ARRAY_REF: NDArray[V] = np.load(PATH / "mol_array.npy")
BOND_ARRAY_REF: NDArray[V] = np.load(PATH / "bond_array.npy")


class Output(NamedTuple):
    mol: list[Molecule]
    bulk: pd.Series
    bulk_ref: pd.Series


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
        "default": dict(),
    }
    IDS = IDS_MAPPING.keys()
    PARAMS = enumerate(IDS_MAPPING.values())

    @staticmethod
    def assert_ex_dict(ex_dict: Mapping[str, AssertionError], i: int) -> None:
        """Evaluate the content of **ex_dict** and raise if it contains any exceptions."""
        __traceback_hide__ = True  # noqa: F841
        if not len(ex_dict):
            return None

        ex = next(iter(ex_dict.values()))
        failed = pformat(list(ex_dict.keys()))
        msg = f"Failed for {len(ex_dict)}/{i + 1} molecules\n{failed}"
        raise AssertionError(msg) from ex

    @pytest.fixture(scope="class", autouse=True, ids=IDS, params=PARAMS, name="output")
    def generate_output(self, request: _pytest.fixtures.SubRequest) -> Output:
        """Generate the test output of :func:`fast_bulk_workflow`."""
        i, kwargs = request.param
        mol_list, v_bulk = fast_bulk_workflow(SMILES, **kwargs)
        return Output(mol_list, v_bulk, FAST_BULK_REF[i])

    def test_mol_properties(self, output: Output) -> None:
        """Validate the properties of the output molecules."""
        ex_dict = {}
        for i, mol in enumerate(output.mol):
            smiles: None | str = mol.properties.get("smiles")
            anchor: None | str = mol.properties.get("anchor")
            name = smiles if smiles is not None else f"mol {i}"
            try:
                assertion.isinstance(smiles, str, message=f"{name} smiles")
                assertion.isinstance(anchor, str, message=f"{name} anchor")
            except AssertionError as ex:
                ex_dict[name] = ex
        self.assert_ex_dict(ex_dict, i)

    def test_mol_atoms(self, output: Output) -> None:
        """Validate the Cartesian coordinates of the output molecules."""
        ex_dict = {}
        for i, mol in enumerate(output.mol):
            name = mol.properties.get("smiles", f"mol {i}")
            try:
                np.testing.assert_allclose(
                    actual=cdist(mol, mol),
                    desired=MOL_ARRAY_REF["dist_mat"][i, :len(mol), :len(mol)],
                    err_msg=f"{name} coordinates\n",
                )
                np.testing.assert_array_equal(
                    x=[at.symbol for at in mol],
                    y=MOL_ARRAY_REF["symbol"][i, :len(mol)],
                    err_msg=f"{name} symbols\n",
                )
            except AssertionError as ex:
                ex_dict[name] = ex
        self.assert_ex_dict(ex_dict, i)

    def test_mol_bonds(self, output: Output) -> None:
        """Validate the bonds and bond orders of the output molecules."""
        ex_dict = {}
        bond_dtype = [("atom1", "i8"), ("atom2", "i8")]
        for i, mol in enumerate(output.mol):
            name = mol.properties.get("smiles", f"mol {i}")
            try:
                mol.set_atoms_id()
                np.testing.assert_array_equal(
                    x=np.fromiter([(b.atom1.id, b.atom2.id) for b in mol.bonds], dtype=bond_dtype),
                    y=BOND_ARRAY_REF[["atom1", "atom2"]][i, :len(mol.bonds)],
                    err_msg=f"{name} bonds\n",
                )
                np.testing.assert_allclose(
                    actual=[b.order for b in mol.bonds],
                    desired=BOND_ARRAY_REF["order"][i, :len(mol.bonds)],
                    err_msg=f"{name} bond orders\n",
                )
            except AssertionError as ex:
                ex_dict[name] = ex
            finally:
                mol.unset_atoms_id()
        self.assert_ex_dict(ex_dict, i)

    @pytest.mark.xfail(reason="FIXME")
    def test_bulk(self, output: Output) -> None:
        """Validate the output bulkiness values."""
        np.testing.assert_allclose(output.bulk, output.bulk_ref)
