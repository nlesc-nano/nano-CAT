"""Tests for :mod:`nanoCAT.bde.dissociate_xyn`."""

from pathlib import Path
from typing import Generator

import numpy as np

from scm.plams import readpdb, Molecule, MoleculeError
from assertionlib import assertion

from nanoCAT.bde.dissociate_xyn import _lig_mapping, MolDissociater, dissociate_ligand


def _get_idx_iter(mol: Molecule) -> Generator[int, None, None]:
    for i, at in enumerate(mol, 1):
        if at.properties.charge == -1:
            at.properties.anchor = True
            yield i


PATH = Path('tests') / 'test_files' / 'bde'
with open(PATH / 'mol.pdb', 'r') as f:
    MOL: Molecule = readpdb(f)

LIG_IDX = np.fromiter(_get_idx_iter(MOL), dtype=int)
LIG_IDX.setflags(write=False)
CORE_IDX = np.array([58, 61, 63, 70, 72, 75])
CORE_IDX.setflags(write=False)


def test_lig_mapping() -> None:
    """Tests for :func:`nanoCAT.bde.dissociate_xyn._lig_mapping`."""
    lig_mapping = _lig_mapping(MOL, LIG_IDX)
    ref = {
        127: [124, 125, 126, 127, 128, 129, 130],
        134: [131, 132, 133, 134, 135, 136, 137],
        141: [138, 139, 140, 141, 142, 143, 144],
        148: [145, 146, 147, 148, 149, 150, 151],
        155: [152, 153, 154, 155, 156, 157, 158],
        162: [159, 160, 161, 162, 163, 164, 165],
        169: [166, 167, 168, 169, 170, 171, 172],
        176: [173, 174, 175, 176, 177, 178, 179],
        183: [180, 181, 182, 183, 184, 185, 186],
        190: [187, 188, 189, 190, 191, 192, 193],
        197: [194, 195, 196, 197, 198, 199, 200],
        204: [201, 202, 203, 204, 205, 206, 207],
        211: [208, 209, 210, 211, 212, 213, 214],
        218: [215, 216, 217, 218, 219, 220, 221],
        225: [222, 223, 224, 225, 226, 227, 228],
        232: [229, 230, 231, 232, 233, 234, 235],
        239: [236, 237, 238, 239, 240, 241, 242],
        246: [243, 244, 245, 246, 247, 248, 249],
        253: [250, 251, 252, 253, 254, 255, 256],
        260: [257, 258, 259, 260, 261, 262, 263],
        267: [264, 265, 266, 267, 268, 269, 270],
        274: [271, 272, 273, 274, 275, 276, 277],
        281: [278, 279, 280, 281, 282, 283, 284],
        288: [285, 286, 287, 288, 289, 290, 291],
        295: [292, 293, 294, 295, 296, 297, 298],
        302: [299, 300, 301, 302, 303, 304, 305]
    }
    assertion.eq(lig_mapping, ref)


def test_get_pairs_closest() -> None:
    """Tests for :meth:`MolDissociater.get_pairs_closest`."""
    dissociate = MolDissociater(MOL, CORE_IDX, ligand_count=2)
    pair1 = dissociate.get_pairs_closest(LIG_IDX, n_pairs=1)
    pair2 = dissociate.get_pairs_closest(LIG_IDX, n_pairs=2)
    pair3 = dissociate.get_pairs_closest(LIG_IDX, n_pairs=3)

    dissociate.ligand_count = 4
    pair4 = dissociate.get_pairs_closest(LIG_IDX, n_pairs=1)
    pair5 = dissociate.get_pairs_closest(LIG_IDX, n_pairs=2)
    pair6 = dissociate.get_pairs_closest(LIG_IDX, n_pairs=3)

    assertion.assert_(dissociate.get_pairs_closest, LIG_IDX, 0, exception=ValueError)

    pairs = (pair1, pair2, pair3, pair4, pair5, pair6)
    for i, pair in enumerate(pairs, 1):
        ref = np.load(PATH / f'get_pairs_closest_{i}.npy')
        np.testing.assert_array_equal(pair, ref)


def test_get_pairs_distance() -> None:
    """Tests for :meth:`MolDissociater.get_pairs_distance`."""
    dissociate = MolDissociater(MOL, CORE_IDX, ligand_count=2)
    pair1 = dissociate.get_pairs_distance(LIG_IDX, max_dist=5.0)
    pair2 = dissociate.get_pairs_distance(LIG_IDX, max_dist=7.5)

    dissociate.ligand_count = 4
    pair3 = dissociate.get_pairs_distance(LIG_IDX, max_dist=7.5)
    pair4 = dissociate.get_pairs_distance(LIG_IDX, max_dist=10.0)

    assertion.assert_(dissociate.get_pairs_distance, LIG_IDX, 0, exception=ValueError)
    assertion.assert_(dissociate.get_pairs_distance, LIG_IDX, 1.0, exception=MoleculeError)
    dissociate.ligand_count = 99
    assertion.assert_(dissociate.get_pairs_distance, LIG_IDX, 5.0, exception=MoleculeError)

    pairs = (pair1, pair2, pair3, pair4)
    for i, pair in enumerate(pairs, 1):
        ref = np.load(PATH / f'get_pairs_distance_{i}.npy')
        np.testing.assert_array_equal(pair, ref)


def test_remove_bulk() -> None:
    """Tests for :meth:`MolDissociater.remove_bulk`."""
    core_idx = (i for i, at in enumerate(MOL, 1) if at.symbol == 'Cd')
    dissociate = MolDissociater(MOL, core_idx, ligand_count=2)
    dissociate.remove_bulk()

    ref = np.load(PATH / 'remove_bulk.npy')
    np.testing.assert_array_equal(dissociate.core_idx, ref)


def test_combinations() -> None:
    """Tests for :meth:`MolDissociater.combinations`."""
    lig_mapping = _lig_mapping(MOL, LIG_IDX)
    dissociate = MolDissociater(MOL, CORE_IDX, ligand_count=2)
    cl_pairs = dissociate.get_pairs_closest(LIG_IDX, n_pairs=1)

    combinations = dissociate.combinations(cl_pairs, lig_mapping)
    ref = {
         (frozenset([58]), frozenset([243, 244, 245, 246, 247, 248, 249, 285, 286, 287, 288, 289, 290, 291])),  # noqa
         (frozenset([61]), frozenset([243, 244, 245, 246, 247, 248, 249, 229, 230, 231, 232, 233, 234, 235])),  # noqa
         (frozenset([63]), frozenset([229, 230, 231, 232, 233, 234, 235, 159, 160, 161, 162, 163, 164, 165])),  # noqa
         (frozenset([70]), frozenset([285, 286, 287, 288, 289, 290, 291, 243, 244, 245, 246, 247, 248, 249])),  # noqa
         (frozenset([72]), frozenset([159, 160, 161, 162, 163, 164, 165, 285, 286, 287, 288, 289, 290, 291])),  # noqa
         (frozenset([75]), frozenset([159, 160, 161, 162, 163, 164, 165, 229, 230, 231, 232, 233, 234, 235]))   # noqa
    }

    assertion.eq(combinations, ref)


def test_call() -> None:
    """Tests for :meth:`MolDissociater.__call__`."""
    mol_iterator = dissociate_ligand(MOL, lig_count=2, core_index=CORE_IDX)
    filename = str(PATH / '{}.pdb')
    for mol in mol_iterator:
        name = mol.properties.core_topology
        mol_ref = readpdb(filename.format(name))
        xyz = mol.as_array()
        ref = mol_ref.as_array()
        np.testing.assert_allclose(xyz, ref)


def test_call_no_core() -> None:
    """Tests for :meth:`MolDissociater.__call__` where `core == lig`."""
    lig_idx = [148, 190, 224, 294]
    dissociate = MolDissociater(MOL, lig_idx, ligand_count=1)
    pair1 = dissociate.get_pairs_closest(LIG_IDX, n_pairs=1)
    pair2 = dissociate.get_pairs_closest(LIG_IDX, n_pairs=2)

    dissociate.ligand_count = 2
    pair3 = dissociate.get_pairs_closest(LIG_IDX, n_pairs=1)
    pair4 = dissociate.get_pairs_closest(LIG_IDX, n_pairs=2)

    pairs = (pair1, pair2, pair3, pair4)
    for i, pair in enumerate(pairs, 1):
        ref = np.load(PATH / f'call_no_core_{i}.npy')
        np.testing.assert_array_equal(pair, ref)

    mol_iterator = dissociate_ligand(MOL, lig_count=1, core_index=lig_idx)
    filename = str(PATH / '{}.pdb')
    for mol in mol_iterator:
        name = mol.properties.core_topology
        mol_ref = readpdb(filename.format(name))
        xyz = mol.as_array()
        ref = mol_ref.as_array()
        np.testing.assert_allclose(xyz, ref)
