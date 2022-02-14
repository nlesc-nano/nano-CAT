from __future__ import annotations

import os
import shutil
from pathlib import Path
from collections.abc import Sequence, Generator
from typing import Any

import numpy as np
import pytest
import yaml
from scm.plams import readpdb, Settings, Molecule

from CAT.base import prep
from CAT.workflows import CONE_ANGLE
from nanoCAT.cone_angle import get_cone_angle

PATH = Path("tests") / "test_files"
CONE_ANGLE_PATH = PATH / "cone_angle"

MOL = readpdb(PATH / "CCCC[O-]@O5.pdb")
MOL_AMMONIUM = Molecule(PATH / "pentyl_ammonium.xyz")
MOL_AMMONIUM.guess_bonds()


class TestConeAngle:
    DIST_PARAMS = {
        "default": (0, 82),
        "scalar": (5, 30),
        "vector": (range(6), [82, 60, 47, 38, 33, 30]),
    }

    @pytest.mark.parametrize("dist,ref", DIST_PARAMS.values(), ids=DIST_PARAMS)
    def test_distance(self, dist: float | Sequence[float], ref: float | Sequence[float]) -> None:
        out = get_cone_angle(MOL, 4, surface_dist=dist)
        np.testing.assert_allclose(out, ref, rtol=0, atol=1)

    RAISE_PARAMS = {
        "anchor_type": ({"anchor": 1.0}, TypeError),
        "dist_ndim": ({"anchor": 4, "surface_dist": [[1.0]]}, ValueError),
    }

    @pytest.mark.parametrize("kwargs,exc_type", RAISE_PARAMS.values(), ids=RAISE_PARAMS)
    def test_raise(self, kwargs: dict[str, Any], exc_type: type[Exception]) -> None:
        with pytest.raises(exc_type):
            get_cone_angle(MOL, **kwargs)

    REMOVE_H_PARAMS = {
        "default": (False, (220, 232)),
        "remove_hydrogens": (True, 81),
    }

    @pytest.mark.parametrize("remove_h,ref", REMOVE_H_PARAMS.values(), ids=REMOVE_H_PARAMS)
    def test_remove_hydrogens(self, remove_h: bool, ref: float | tuple[float, float]) -> None:
        out = get_cone_angle(MOL_AMMONIUM, 5, remove_anchor_hydrogens=remove_h)

        # NOTE: The pentyl ammonium system seems to converge to one of two cone-angles
        if isinstance(ref, tuple):
            try:
                np.testing.assert_allclose(out, ref[0], rtol=0, atol=1)
            except AssertionError:
                np.testing.assert_allclose(out, ref[1], rtol=0, atol=1)
        else:
            np.testing.assert_allclose(out, ref, rtol=0, atol=1)


class TestWorkflow:
    with open(PATH / "cone_angle_setting.yaml", "r") as f:
        SETTINGS = Settings(yaml.load(f, Loader=yaml.SafeLoader))

    @pytest.fixture(scope="class", autouse=True)
    def setup_cat(self) -> Generator[None, None, None]:
        if not os.path.isdir(CONE_ANGLE_PATH):
            os.mkdir(CONE_ANGLE_PATH)
        yield None
        for i in os.listdir(CONE_ANGLE_PATH):
            shutil.rmtree(CONE_ANGLE_PATH / i, ignore_errors=True)

    def test_pass(self) -> None:
        s = self.SETTINGS.copy()
        *_, ligand_df = prep(s)
        ref = [61, 81, 85]
        np.testing.assert_allclose(ligand_df[CONE_ANGLE], ref, rtol=0, atol=1)
