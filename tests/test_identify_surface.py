"""Tests for :mod:`nanoCAT.bde.identify_surface`."""

from pathlib import Path

import numpy as np

from scm.plams import Molecule
from assertionlib import assertion

from nanoCAT.bde.identify_surface import identify_surface

PATH = Path('tests') / 'test_files'
MOL = Molecule(PATH / 'Cd360Se309.xyz')


def test_identify_surface() -> None:
    """Tests for :func:`nanoCAT.bde.guess_core_dist.guess_core_core_dist`."""
    idx_superset = np.array([i for i, atom in enumerate(MOL) if atom.symbol == 'Cd'])
    xyz = np.array(MOL)[idx_superset]

    idx_subset = idx_superset[identify_surface(xyz)]
    ref = [3, 4, 5, 6, 13, 14, 16, 23, 26, 32, 33, 34, 36, 41, 46, 48, 53, 54, 55, 56, 57, 58, 62,
           63, 65, 68, 69, 74, 77, 79, 80, 82, 83, 86, 92, 93, 94, 95, 96, 99, 101, 105, 106, 111,
           112, 113, 118, 119, 120, 126, 128, 129, 135, 138, 143, 144, 146, 147, 153, 155, 156, 157,
           158, 159, 160, 161, 162, 163, 167, 168, 170, 171, 172, 173, 174, 175, 176, 177, 179, 181,
           183, 186, 187, 188, 190, 191, 192, 193, 196, 199, 200, 201, 205, 206, 210, 212, 217, 221,
           226, 227, 228, 229, 234, 235, 237, 238, 239, 240, 241, 242, 243, 244, 247, 250, 252, 253,
           254, 255, 258, 260, 263, 264, 265, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277,
           283, 285, 288, 289, 290, 291, 294, 295, 297, 300, 301, 302, 303, 305, 306, 307, 308, 309,
           311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 325, 327, 328, 330, 331,
           333, 335, 337, 338, 340, 348, 349, 353, 354, 355, 356, 357, 358, 359]
    np.testing.assert_array_equal(idx_subset, ref)

    assertion.assert_(identify_surface, MOL, max_dist=1, exception=ValueError)
