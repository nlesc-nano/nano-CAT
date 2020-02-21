"""Tests for :mod:`nanoCAT.recipes.dissociate_surface`."""

from itertools import chain
from pathlib import Path

import numpy as np

from scm.plams import Molecule, PTError
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

        [320, 319],

        [[320, 319],
         [158, 57],
         [156, 155]]
    )

    at_idx_iter = iter([
        319,
        319,
        320, 319,
        320, 319, 158, 57, 156, 155
    ])

    mol_iter = chain.from_iterable(dissociate_surface(MOL, i) for i in idx_tup)
    for i, mol in zip(at_idx_iter, mol_iter):
        assertion.contains(np.asarray(mol), XYZ[i], invert=True)

    assertion.assert_(next, dissociate_surface(MOL, i, k=0), exception=ValueError)
    assertion.assert_(next, dissociate_surface(MOL, i, k=999), exception=ValueError)
    assertion.assert_(next, dissociate_surface(MOL, i, lig_count=999), exception=ValueError)
    assertion.assert_(next, dissociate_surface(MOL, i, lig_count=-1), exception=ValueError)
    assertion.assert_(next, dissociate_surface(MOL, i, symbol='bob'), exception=PTError)
    assertion.assert_(next, dissociate_surface(MOL, i, symbol=999), exception=PTError)
    assertion.assert_(next, dissociate_surface(MOL, i, symbol=9.5), exception=TypeError)
