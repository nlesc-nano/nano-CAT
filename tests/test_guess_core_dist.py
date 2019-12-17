"""Tests for :mod:`nanoCAT.bde.guess_core_dist`."""

from pathlib import Path

from scm.plams import readpdb, Molecule
from assertionlib import assertion

from nanoCAT.bde.guess_core_dist import guess_core_core_dist

PATH = Path('tests') / 'test_files'
with open(PATH / 'mol.pdb', 'r') as f:
    MOL: Molecule = readpdb(f)


def test_guess_core_core_dist() -> None:
    """Tests for :func:`nanoCAT.bde.guess_core_dist.guess_core_core_dist`."""
    dist = guess_core_core_dist(MOL, 'Cd')
    ref = 5.175623211939212
    assertion.allclose(dist, ref)
