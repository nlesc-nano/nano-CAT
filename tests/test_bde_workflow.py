from __future__ import annotations

import os
import shutil
import textwrap
from collections.abc import Generator
from typing import Any
from pathlib import Path
from distutils.spawn import find_executable

import yaml
import pytest
import numpy as np
import pandas as pd
from assertionlib import assertion
from nanoutils import UniqueLoader
from scm.plams import readpdb, Settings, Cp2kJob

from CAT.base import validate_input
from CAT.settings_dataframe import SettingsDataFrame
from CAT.workflows import MOL
from nanoCAT.bde import init_bde

PATH = Path("tests") / "test_files"
BASE_SETTINGS = Settings(yaml.load(textwrap.dedent("""
    path: tests/test_files
    input_cores:
        -   Br5ClCs8Pb.xyz
    input_ligands:
        -   CO
    optional:
        database:
            read: False
            write: False
        qd:
            dissociate:
                core_atom: null
                core_index: null
                lig_count: 1

                keep_files: True
                xyn_pre_opt: False
                job1: Cp2kJob
                s1:
                    input:
                        motion:
                            geo_opt:
                                type: minimization
                                optimizer: bfgs
                                max_iter: 10
                                max_force: 1e-03
                        force_eval:
                                dft:
                                    basis_set_file_name: BASIS_MOLOPT
                                    potential_file_name: GTH_POTENTIALS
                                    xc:
                                        xc_functional pbe: {}
                                    scf:
                                        eps_scf: 1e-05
                                        max_scf: 20
                                subsys:
                                    cell:
                                        abc: 8.00  8.00  8.00
                                        periodic: None
                                    kind H:
                                        basis_set: DZVP-MOLOPT-SR-GTH
                                        potential: GTH-PBE
                                    kind Cl:
                                        basis_set: DZVP-MOLOPT-SR-GTH
                                        potential: GTH-PBE
                job2: null
                s2: null
""").strip(), Loader=UniqueLoader))


def construct_df() -> SettingsDataFrame:
    # Construct the settings
    settings = BASE_SETTINGS.copy()
    validate_input(settings)
    settings.optional.database.db = None
    settings.optional.qd.dissociate.xyn_opt = False

    # Set all quantum dot properties
    qd = readpdb(PATH / "[HCl]2.pdb")
    qd.properties.name = "[HCl]2"
    qd.properties.job_path = []
    qd[4].properties.anchor = True

    # Construct the dataframe
    columns = pd.MultiIndex.from_tuples([MOL], names=['index', 'sub index'])
    index = pd.MultiIndex.from_tuples(
        [('H2Cl2', '1', 'Cl[-]', 'Cl1')],
        names=['core', 'core anchor', 'ligand smiles', 'ligand anchor'],
    )
    df = SettingsDataFrame(index=index, columns=columns, settings=settings)
    df[MOL] = [qd]
    df.settings = Settings(df.settings)  # unfreeze the settings
    return df


@pytest.mark.skipif(find_executable("cp2k.popt") is None, reason="requires CP2K")
@pytest.mark.slow
class TestBDEWorkflow:
    PARAMS = dict(
        core_index={"core_index": [3]},
        core_atom={"core_atom": "H", "lig_core_dist": 2.0},
        freq={"core_index": [3], "job2": Cp2kJob,
              "s2": BASE_SETTINGS.optional.qd.dissociate.s1.copy()},
        qd_opt={"core_index": [3], "qd_opt": True},
    )

    @pytest.fixture(scope="class", autouse=True)
    def clear_dirs(self) -> Generator[None, None, None]:
        """Teardown script for deleting directies."""
        os.mkdir(PATH / "qd")
        yield None
        shutil.rmtree(PATH / "database", ignore_errors=True)
        shutil.rmtree(PATH / "ligand", ignore_errors=True)
        shutil.rmtree(PATH / "qd", ignore_errors=True)

    @pytest.mark.parametrize("kwargs", PARAMS.values(), ids=PARAMS)
    def test_pass(self, kwargs: dict[str, Any], clear_dirs: None) -> None:
        qd_df = construct_df()
        qd_df.settings.optional.qd.dissociate.update(kwargs)
        init_bde(qd_df)

        bde = qd_df["BDE dE", "0"].iloc[0]
        assertion.len_eq(qd_df["BDE dE"].columns, 1)
        assertion.assert_(np.isfinite, bde)

        if "job2" in kwargs:
            bde_gibbs = qd_df["BDE dG", "0"].iloc[0]
            assertion.len_eq(qd_df["BDE dG"].columns, 1)
            assertion.assert_(np.isfinite, bde_gibbs)
