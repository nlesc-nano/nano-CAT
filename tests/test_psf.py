"""Tests for :class:`nanoCAT.ff.psf.PSFContainer`."""

import os
from os.path import join
from tempfile import TemporaryFile
from itertools import zip_longest

import numpy as np
import pandas as pd

from CAT.assertion_functions import assert_instance, assert_eq, assert_exception, assert_isin
from nanoCAT.ff.psf import PSFContainer

PATH: str = join('tests', 'test_files', 'psf')
PATH = '/Users/basvanbeek/Documents/GitHub/nano-CAT/tests/test_files/psf'
PSF: PSFContainer = PSFContainer.read(join(PATH, 'mol.psf'))


def test_write() -> None:
    """Tests for :meth:`PSFContainer.write`."""
    filename1 = join(PATH, 'mol.psf')
    filename2 = join(PATH, 'tmp.psf')

    try:
        PSF.write(filename2)
        with open(filename1) as f1, open(filename2) as f2:
            for i, j in zip_longest(f1, f2):
                assert_eq(i, j)
    finally:
        if os.path.isfile(filename2):
            os.remove(filename2)

    with open(filename1, 'rb') as f1, TemporaryFile() as f2:
        PSF.write(f2, encoding='utf-8')
        f2.seek(0)
        for i, j in zip_longest(f1, f2):
            assert_eq(i, j)


def test_update_atom_charge():
    """Tests for :meth:`PSFContainer.update_atom_charge`."""
    psf = PSF.copy()
    psf.update_atom_charge('C2O3', -5.0)
    condition = psf.atom_type == 'C2O3'

    assert (psf.charge[condition] == -5.0).all()
    assert_exception(ValueError, psf.update_atom_charge, 'C2O3', 'bob')
    assert_exception(KeyError, psf.update_atom_charge, 'bob', -5.0)


def test_update_atom_type():
    """Tests for :meth:`PSFContainer.update_atom_type`."""
    psf = PSF.copy(deep=True)
    psf.update_atom_type('C2O3', 'C8')

    assert_isin('C8', psf.atom_type)
    assert_exception(KeyError, psf.update_atom_charge, 'bob', 'C2O3')
