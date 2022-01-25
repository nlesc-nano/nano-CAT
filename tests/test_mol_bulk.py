"""Tests for :mod:`nanoCAT.mol_bulk`."""

import os
import shutil
from pathlib import Path
from typing import Generator

import yaml
import pytest
import numpy as np
from scm.plams import readpdb, Settings
from assertionlib import assertion

from CAT.base import prep
from nanoCAT.mol_bulk import get_lig_radius, get_V

PATH = Path('tests') / 'test_files'
BULK_PATH = PATH / "bulk"
MOL = readpdb(PATH / 'hexanoic_acid.pdb')


class TestMain:
    with open(PATH / "bulk_settings.yaml", "r") as f:
        SETTINGS = Settings(yaml.load(f, Loader=yaml.SafeLoader))

    @pytest.fixture(scope="function", autouse=True)
    def setup_cat(self) -> Generator[None, None, None]:
        if not os.path.isdir(BULK_PATH):
            os.mkdir(BULK_PATH)
        yield None
        for i in os.listdir(BULK_PATH):
            shutil.rmtree(BULK_PATH / i, ignore_errors=True)

    @pytest.mark.parametrize("d", [None, 5.0, range(3, 7)], ids=["None", "0d", "1d"])
    @pytest.mark.parametrize("h_lim", [None, 5.0], ids=["None", "0d"])
    def test_pass(self, d, h_lim, setup_cat) -> None:
        s = self.SETTINGS.copy()
        s.optional.qd.bulkiness = {"d": d, "h_lim": h_lim}
        qd_df, *_ = prep(s)


class TestGetV:
    radius, height = get_lig_radius(MOL, anchor=6)
    radius.setflags(write=False)
    height.setflags(write=False)

    @pytest.mark.parametrize("d", [None, np.float64(5), np.arange(3, 7)], ids=["None", "0d", "1d"])
    @pytest.mark.parametrize("h_lim", [None, 5.0], ids=["None", "0d"])
    def test_pass(self, d, h_lim) -> None:
        out = get_V(self.radius, self.height, d=d, angle=None, h_lim=h_lim)
        ndim_1 = getattr(d, "ndim", 0) == 1
        if ndim_1:
            assertion.eq(out.shape, (4,))
        else:
            assertion.eq(out.shape, ())
        assertion.assert_(np.isreal, out, post_process=np.all)
