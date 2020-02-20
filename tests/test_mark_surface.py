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

    ref = [3, 5, 6, 14, 32, 34, 36, 48, 55, 57, 63, 68, 69, 74, 80, 82, 86, 95, 101, 105, 106, 112,
           126, 128, 129, 135, 138, 143, 155, 157, 159, 162, 167, 168, 170, 171, 172, 173, 175, 177,
           179, 181, 191, 193, 196, 205, 206, 210, 212, 228, 229, 235, 238, 240, 242, 243, 252, 253,
           254, 255, 258, 260, 263, 264, 265, 268, 269, 270, 272, 273, 276, 285, 289, 291, 297, 301,
           302, 303, 305, 307, 308, 309, 312, 320, 321, 323, 330, 333, 337, 338, 357, 359]
    idx = [i for i, at in enumerate(mol_new) if at.atnum == 1]
    np.testing.assert_array_equal(idx, ref)

    assertion.assert_(replace_surface, symbol='I', exception=MoleculeError)
    assertion.assert_(replace_surface, symbol='bob', exception=PTError)
