"""Tests for :mod:`nanoCAT.mol_bulk`."""

from os.path import join

import numpy as np

from scm.plams import Molecule, readpdb

from CAT.assertion.assertion_manager import assertion
from nanoCAT.mol_bulk import get_cone_angles, get_V

PATH: str = join('tests', 'test_files')
MOL: Molecule = readpdb(join(PATH, 'hexanoic_acid.pdb'))
MOL[7].properties.anchor = True


def test_get_cone_angles() -> None:
    """Tests for :func:`nanoCAT.mol_bulk.get_cone_angles`."""
    ref = np.load(join(PATH, 'get_cone_angles.npy'))
    angles = get_cone_angles(MOL)
    np.testing.assert_allclose(angles, ref)


def test_get_V() -> None:
    """Tests for :func:`nanoCAT.mol_bulk.get_V`."""
    angles = get_cone_angles(MOL)
    V_bulk = get_V(angles)
    ref = 1.63215388
    assertion.allclose(ref, V_bulk)
