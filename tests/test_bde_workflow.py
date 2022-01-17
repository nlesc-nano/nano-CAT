from __future__ import annotations

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
from scm.plams import readpdb, Settings

from CAT.base import validate_input
from CAT.settings_dataframe import SettingsDataFrame
from CAT.workflows import MOL
from nanoCAT.bde import init_bde

PATH = Path("tests") / "test_files"


def construct_df() -> SettingsDataFrame:
    # Construct the settings
    settings = Settings(yaml.load(textwrap.dedent("""
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
                    core_core_dist: 2.0
                    lig_count: 1

                    keep_files: True
                    xyn_pre_opt: False
                    job1: Cp2kJob
                    s1:
                        input:
                            FORCE_EVAL:
                                    DFT:
                                        BASIS_SET_FILE_NAME: BASIS_MOLOPT
                                        POTENTIAL_FILE_NAME: GTH_POTENTIALS
                                        XC:
                                            XC_FUNCTIONAL pbe: {}
                                    SUBSYS:
                                        CELL:
                                            ABC: 7.00  7.00  7.00
                                            PERIODIC: None
                                        KIND H:
                                            BASIS_SET: DZVP-MOLOPT-SR-GTH-q1
                                            POTENTIAL: GTH-PBE-q1
                                        KIND Cl:
                                            BASIS_SET: DZVP-MOLOPT-SR-GTH-q7
                                            POTENTIAL: GTH-PBE-q7
                    job2: null
                    s2: null
    """).strip(), Loader=UniqueLoader))
    validate_input(settings)
    settings.optional.database.db = None
    settings.optional.qd.dissociate.xyn_opt = False

    # Set all quantum dot properties
    qd = readpdb(PATH / "qd" / "[HCl]2.pdb")
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
class TestBDEWorkflow:
    PARAMS = dict(
        core_index=("core_index", [3]),
    )

    @pytest.fixture(scope="function", autouse=True)
    def clear_db(self) -> Generator[None, None, None]:
        """Teardown script for deleting directies."""
        yield None
        shutil.rmtree(PATH / "database", ignore_errors=True)
        shutil.rmtree(PATH / "qd" / "bde", ignore_errors=True)

    @pytest.mark.parametrize("key,value", PARAMS.values(), ids=PARAMS.keys())
    def test_core_index(self, key: str, value: Any, clear_db: None) -> None:
        qd_df = construct_df()
        qd_df.settings.optional.qd.dissociate[key] = value
        init_bde(qd_df)

        bde = qd_df["BDE dE", "0"].iloc[0]
        assertion.len_eq(qd_df["BDE dE"].columns, 1)
        assertion.assert_(np.isfinite, bde)
