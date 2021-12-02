"""Tests for :mod:`nanoCAT.recipes.coordination_number`."""

from pathlib import Path

import numpy as np

from scm.plams import Molecule
from assertionlib import assertion

from nanoCAT.recipes import get_coordination_number

PATH = Path('tests') / 'test_files'
MOL = Molecule(PATH / 'Cd68Se55.xyz')


def test_coordination_number() -> None:
    """Tests for :func:`nanoCAT.recipes.get_coordination_number`."""

    out_inner = get_coordination_number(MOL, shell='inner')

    out_outer = get_coordination_number(MOL, shell='outer', d_outer=5.2)

    ref_inner = {'Cd': {3: [30, 31, 34, 35, 36, 46, 47, 52, 53, 54, 57, 58, 59, 60, 63, 64, 65, 67,
                            69, 70, 71, 75, 78, 79],
                        4: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 33, 37, 38,
                            39, 40, 41, 42, 43, 44, 45, 48, 49, 50, 51, 55, 56, 61, 62, 66, 68, 72,
                            73, 74, 76, 77, 80, 81]},
                 'Se': {3: [82, 84, 87, 88, 90, 91, 92, 95, 96, 97, 99, 100, 101, 102, 103, 106,
                            107, 108, 118, 119, 120, 121, 122, 123],
                        4: [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 83, 85, 86, 89, 93,
                            94, 98, 104, 105, 109, 110, 111, 112, 113, 114, 115, 116, 117]},
                 'Cl': {2: [124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
                            138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]}}

    ref_outer = {'Cd': {6: [33, 59, 65, 66], 7: [5, 6, 8, 13, 39, 77],
                        8: [30, 31, 32, 34, 35, 36, 46, 47, 52, 53, 54, 57, 58, 60, 63, 64, 67, 69,
                            70, 71, 75, 78, 79, 81],
                        9: [4, 11, 40, 41, 48, 49, 50, 51, 55, 73, 76, 80],
                        10: [7, 10, 12, 15, 42, 44, 45, 56],
                        11: [2, 16, 43, 68], 12: [1, 3, 9, 14, 37, 38, 61, 62, 72, 74]},
                 'Se': {6: [102, 121], 7: [84, 87, 97, 99, 103, 106, 108, 118],
                        8: [88, 90, 91, 92, 95, 96, 100, 101, 107, 119, 120, 122],
                        9: [82, 104, 115, 123], 10: [98, 109, 110], 11: [86, 105],
                        12: [20, 22, 27, 28, 83, 113, 114], 13: [18, 19, 24, 26, 85, 117],
                        14: [17, 21, 25, 29, 89, 93, 94, 111, 112, 116], 16: [23]},
                 'Cl': {7: [124, 128], 9: [127, 132, 133, 136, 144, 148],
                        10: [125, 126, 129, 130, 131, 134, 135, 137, 138, 139, 140, 141, 142, 143,
                             145, 146, 147, 149]}}

    np.testing.assert_equal(out_inner, ref_inner)
    np.testing.assert_equal(out_outer, ref_outer)

    assertion.assert_(get_coordination_number, MOL, shell='bob', exception=ValueError)
    assertion.assert_(get_coordination_number, MOL, shell='outer', d_outer=None,
                      exception=TypeError)
