from __future__ import annotations

import shutil
from pathlib import Path
from collections.abc import Generator
from distutils.spawn import find_executable

import numpy as np
import yaml
import pytest
from scm.plams import Settings
from assertionlib import assertion
from CAT.base import prep

PATH = Path("tests") / "test_files"


@pytest.mark.skipif(find_executable("cp2k.popt") is None, reason="requires CP2K")
@pytest.mark.slow
class TestMDASA:
    SETTINGS = Settings(yaml.load("""\
        path: tests/test_files

        input_cores:
            - Cd68Se55.xyz

        input_ligands:
            - 'CCCCCCCCC(=O)O'

        optional:
            core:
                dummy: Cl

            ligand:
                optimize: True
                split: True
                functional_groups: 'O(C=O)[H]'

            qd:
                dirname: QD
                optimize: False
                activation_strain:
                    use_ff: True
                    md: True
                    dump_csv: True
                    job1: Cp2kJob
                    s1:
                        input:
                            motion:
                                md:
                                    steps:
                                        100
                    shift_cutoff: True
                    el_scale14: 1.0
                    lj_scale14: 1.0
                    distance_upper_bound: 10.0

            forcefield:
                charge:
                    keys: [input, force_eval, mm, forcefield, charge]
                    Cd: 0.9768
                    Se: -0.9768
                    O2D2: -0.4704
                    C2O3: 0.4524
                epsilon:
                    unit: kjmol
                    keys: [input, force_eval, mm, forcefield, nonbonded, lennard-jones]
                    Cd Cd: 0.3101
                    Se Se: 0.4266
                    Cd Se: 1.5225
                    Cd O2D2: 1.8340
                    Se O2D2: 1.6135
                sigma:
                    unit: nm
                    keys: [input, force_eval, mm, forcefield, nonbonded, lennard-jones]
                    Cd Cd: 0.1234
                    Se Se: 0.4852
                    Cd Se: 0.2940
                    Cd O2D2: 0.2471
                    Se O2D2: 0.3526
    """, Loader=yaml.SafeLoader))

    @pytest.fixture(scope="class", autouse=True)
    def clear_dirs(self) -> Generator[None, None, None]:
        """Teardown script for deleting directies."""
        yield None
        shutil.rmtree(PATH / "database", ignore_errors=True)
        shutil.rmtree(PATH / "ligand", ignore_errors=True)
        shutil.rmtree(PATH / "qd", ignore_errors=True)

    def test_pass(self) -> None:
        df, *_ = prep(self.SETTINGS.copy())

        values = df["ASA"].values
        assertion.eq(values.dtype.type, np.float64)
        assertion.assert_(np.isreal, values, post_process=np.all)
        assertion.eq(values.shape, (3, 1))
