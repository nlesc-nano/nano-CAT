"""Tests for :mod:`nanoCAT.recipes.replace_surface`."""

from pathlib import Path

import numpy as np

from scm.plams import Molecule, MoleculeError, PTError
from assertionlib import assertion

from nanoCAT.recipes import replace_surface

PATH = Path('tests') / 'test_files'
MOL = Molecule(PATH / 'Cd360Se309.xyz')


def test_replace_surface() -> None:
    """Tests for :func:`replace_surface`."""
    mol_new = replace_surface(MOL, symbol='Cd', symbol_new='H')

    ref = [5, 13, 16, 32, 48, 53, 54, 55, 56, 58, 68, 69, 77, 79, 80, 83, 86, 92, 93, 105, 106,
           112, 113, 118, 119, 129, 135, 138, 143, 155, 157, 159, 160, 170, 171, 172, 176, 177,
           181, 183, 186, 192, 193, 196, 200, 205, 206, 212, 217, 227, 229, 234, 235, 238, 240,
           241, 242, 247, 253, 254, 258, 260, 267, 270, 271, 272, 276, 285, 290, 291, 294, 295,
           300, 301, 307, 313, 315, 319, 321, 323, 325, 328, 331, 335, 337, 340, 348, 349, 353,
           355, 357, 359]

    idx = [i for i, at in enumerate(mol_new) if at.atnum == 1]
    np.testing.assert_array_equal(idx, ref)

    assertion.assert_(replace_surface, MOL, symbol='I', exception=MoleculeError)
    assertion.assert_(replace_surface, MOL, symbol='bob', exception=PTError)
