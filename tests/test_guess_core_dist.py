"""Tests for :mod:`nanoCAT.bde.guess_core_dist`."""

from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from scm.plams import readpdb, Molecule
from assertionlib import assertion

from nanoCAT.bde.guess_core_dist import guess_core_core_dist, get_rdf

PATH = Path('tests') / 'test_files'
with open(PATH / 'mol.pdb', 'r') as f:
    MOL: Molecule = readpdb(f)


def test_get_rdf() -> None:
    """Tests for :func:`nanoCAT.bde.guess_core_dist.get_rdf`."""
    i = [i for i, at in enumerate(MOL) if at.symbol == 'Cd']
    xyz = MOL.as_array()
    dist = cdist(xyz[i], xyz[i])

    ref = np.load(PATH / 'RDF.npy')
    rdf = get_rdf(dist)
    np.testing.assert_allclose(rdf, ref)


def test_guess_core_core_dist() -> None:
    """Tests for :func:`nanoCAT.bde.guess_core_dist.guess_core_core_dist`."""
    dist = guess_core_core_dist(MOL, 'Cd')
    ref = 5.175623211939212
    assertion.allclose(dist, ref)
