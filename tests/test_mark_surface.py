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
    ref1 = [5, 13, 16, 32, 48, 53, 54, 55, 56, 58, 68, 69, 77, 79, 80, 83, 86, 92, 93, 105, 106,
            112, 113, 118, 119, 129, 135, 138, 143, 155, 157, 159, 160, 170, 171, 172, 176, 177,
            181, 183, 186, 192, 193, 196, 200, 205, 206, 212, 217, 227, 229, 234, 235, 238, 240,
            241, 242, 247, 253, 254, 258, 260, 267, 270, 271, 272, 276, 285, 290, 291, 294, 295,
            300, 301, 307, 313, 315, 319, 321, 323, 325, 328, 331, 335, 337, 340, 348, 349, 353,
            355, 357, 359]
    ref2 = [2, 7, 12, 15, 17, 22, 25, 31, 36, 42, 44, 47, 49, 50, 51, 52, 59, 60, 61, 64, 66, 67,
            71, 72, 73, 75, 76, 81, 85, 87, 88, 90, 91, 97, 98, 102, 103, 104, 107, 110, 114, 117,
            121, 125, 130, 134, 137, 148, 149, 150, 151, 152, 154, 164, 165, 178, 180, 182, 184,
            189, 194, 197, 198, 202, 204, 207, 209, 211, 213, 214, 215, 216, 220, 222, 223, 224,
            225, 230, 231, 232, 233, 236, 249, 256, 259, 261, 266, 278, 280, 281, 282, 284, 292,
            293, 296, 298, 299, 304, 310, 326, 329, 332, 339, 350, 351, 352]
    ref3 = [1, 8, 11, 18, 21, 24, 27, 35, 37, 38, 39, 40, 43, 84, 89, 108, 109, 115, 116, 122, 123,
            124, 127, 131, 132, 133, 136, 139, 140, 141, 142, 145, 166, 169, 185, 203, 208, 218,
            245, 246, 248, 251, 257, 279, 287, 324, 336, 341, 342, 343, 344, 345, 346, 347]

    mol_new1 = replace_surface(MOL, symbol='Cd', symbol_new='H')
    idx1 = [i for i, at in enumerate(mol_new1) if at.atnum == 1]
    np.testing.assert_array_equal(idx1, ref1)

    mol_new2 = replace_surface(MOL, symbol='Cd', symbol_new='H', f=1, nth_shell=1)
    idx2 = [i for i, at in enumerate(mol_new2) if at.atnum == 1]
    np.testing.assert_array_equal(idx2, ref2)

    mol_new3 = replace_surface(MOL, symbol='Cd', symbol_new='H', f=1, nth_shell=2)
    idx3 = [i for i, at in enumerate(mol_new3) if at.atnum == 1]
    np.testing.assert_array_equal(idx3, ref3)

    mol_new4 = replace_surface(MOL, symbol='Cd', symbol_new='H', f=1, nth_shell=[1, 2])
    idx4 = [i for i, at in enumerate(mol_new4) if at.atnum == 1]
    np.testing.assert_array_equal(idx4, sorted(ref2 + ref3))

    assertion.isdisjoint(idx1, idx2)
    assertion.isdisjoint(idx1, idx3)
    assertion.isdisjoint(idx2, idx3)

    assertion.assert_(replace_surface, MOL, symbol='I', exception=MoleculeError)
    assertion.assert_(replace_surface, MOL, symbol='Cd', nth_shell=100, exception=MoleculeError)
    assertion.assert_(replace_surface, MOL, symbol='bob', exception=PTError)
