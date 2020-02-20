"""Tests for :mod:`nanoCAT.recipes.dissociate_surface`."""

from pathlib import Path

import numpy as np

from scm.plams import Molecule, MoleculeError, PTError
from assertionlib import assertion

from nanoCAT.recipes import dissociate_surface

PATH = Path('tests') / 'test_files'
MOL = Molecule(PATH / 'Cd360Se309.xyz')
XYZ = np.array(MOL)


def test_dissociate_surface() -> None:
    """Tests for :func:`dissociate_surface`."""
    idx_tup = (
        319,

        [319],

        [319, 320],

        [[319, 320],
         [158, 57],
         [156, 155]]
    )

    for idx in idx_tup:
        mol_iter = dissociate_surface(MOL, idx)
