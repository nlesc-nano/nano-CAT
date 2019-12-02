"""
nanoCAT.asa
===========

A module related to performing activation strain analyses.

Index
-----
.. currentmodule:: nanoCAT.asa
.. autosummary::
    init_asa
    get_asa_energy

API
---
.. autofunction:: init_asa
.. autofunction:: get_asa_energy

"""

from typing import Optional, Iterable, Tuple, List, Any, Type, Generator
from itertools import combinations

import numpy as np

from scm.plams import Settings, Molecule, Cp2kJob
from scm.plams.core.basejob import Job
import scm.plams.interfaces.molecule.rdkit as molkit

from rdkit.Chem import AllChem

from FOX import get_non_bonded
from CAT.jobs import job_md
from CAT.mol_utils import round_coords
from CAT.workflows.workflow import WorkFlow
from CAT.settings_dataframe import SettingsDataFrame
from CAT.attachment.qd_opt_ff import qd_opt_ff

__all__ = ['init_asa_md']

# Aliases for pd.MultiIndex columns
MOL: Tuple[str, str] = ('mol', '')
JOB_SETTINGS_ASA_MD: Tuple[str, str] = ('job_settings_ASA_MD', '')

UFF = AllChem.UFFGetMoleculeForceField


def init_asa_md(qd_df: SettingsDataFrame) -> None:
    """Initialize the activation-strain analyses (ASA).

    The ASA (RDKit UFF level) is conducted on the ligands in the absence of the core.

    Parameters
    ----------
    |CAT.SettingsDataFrame|
        A dataframe of quantum dots.

    """
    workflow = WorkFlow.from_template(qd_df, name='asa_md')

    # Run the activation strain workflow
    idx = workflow.from_db(qd_df)
    workflow(get_asa_md, qd_df, index=idx)

    # Prepare for results exporting
    qd_df[JOB_SETTINGS_ASA_MD] = workflow.pop_job_settings(qd_df[MOL])
    job_recipe = workflow.get_recipe()
    workflow.to_db(qd_df, index=idx, job_recipe=job_recipe)


def get_asa_md(mol_list: Iterable[Molecule],
               jobs: Tuple[Type[Job], ...],
               settings: Tuple[Settings, ...],
               read_template: bool = True,
               **kwargs: Any) -> np.ndarray:
    r"""Perform an activation strain analyses (ASA).

    The ASA calculates the interaction, strain and total energy.

    Parameters
    ----------
    mol_list : :class:`Iterable<collectionc.abc.Iterable>` [:class:`Molecule`]
        An iterable consisting of PLAMS molecules.

    jobs : :class:`tuple` [|plams.Job|]
        A tuple that may or may not contain |plams.Job| types.
        Will default to RDKits' implementation of UFF if ``None``.

    settings : :class:`tuple` [|plams.Settings|]
        A tuple that may or may not contain job |plams.Settings|.
        Will default to RDKits' implementation of UFF if ``None``.

    \**kwargs : :data:`Any<typing.Any>`
        Further keyword arguments for ensuring signature compatiblity.

    Returns
    -------
    Three :math:`n*3` |np.ndarray|_ [|np.float64|_]
        A 2D array containing :math:`E_{int}`, :math:`E_{strain}` and :math:`E`
        for all *n* molecules in **mol_series**.

    """
    job = jobs[0]
    s = settings[0]
    if job is not Cp2kJob:
        raise ValueError("'jobs' expected '(Cp2kJob,)'")

    E_int_iterator = (i for i in _md_iterator(mol_list, job, s))
    E_int = np.from_iter(E_int_iterator, dtype=float)


def _md_iterator(mol_list: Iterable[Molecule], job: Type[Job],
                 settings: Settings) -> Generator[float, None, None]:
    for mol in mol_list:
        results = mol.job_md(job, settings, ret_results=True)  # Run the MD

        # Manually calculate all inter-ligand, ligand/core & core/core interactions
        df = get_non_bonded(
            results['PROJECT-pos-1.xyz'],
            psf=settings.input.force_eval.subsys.topology.conn_file_name,
            prm=settings.input.force_eval.mm.forcefield.parm_file_name,
            cp2k_settings=settings
        )

        # Set all core/core interactions to 0.0
        symbol_set = {at.symbol for at in mol if at.properties.pdb_info.ResidueName == 'COR'}
        for key in combinations(sorted(symbol_set), r=2):
            df.loc[key] = 0.0

        yield df.values.sum()
