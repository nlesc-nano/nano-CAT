"""Tests for :mod:`nanoCAT.bde.identify_surface`."""

from pathlib import Path

import numpy as np

from scm.plams import readpdb, Molecule
from assertionlib import assertion

from nanoCAT.bde.identify_surface import identify_surface

PATH = Path('tests') / 'test_files'
with open(PATH / 'mol.pdb', 'r') as f:
    MOL: Molecule = readpdb(f)


def test_identify_surface() -> None:
    """Tests for :func:`nanoCAT.bde.guess_core_dist.guess_core_core_dist`."""
    idx_superset = np.array([i for i, atom in enumerate(MOL) if atom.symbol == 'Cd'])
    xyz1 = np.array(MOL)[idx_superset]

    idx_subset = idx_superset[identify_surface(xyz1)]
    ref = [29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
           51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
           73, 74, 75, 76, 77, 78, 79, 80]
    np.testing.assert_array_equal(idx_subset, ref)

    xyz2 = xyz1 * 1000
    assertion.assert_(identify_surface, xyz2, exception=ValueError)
