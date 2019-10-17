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

from os.path import join
from shutil import rmtree
from typing import Optional, Union, Iterable, Tuple, List, Any, Type

import numpy as np

from scm.plams import Settings, Molecule, finish
from scm.plams.core.basejob import Job
import scm.plams.interfaces.molecule.rdkit as molkit

import rdkit
from rdkit.Chem import AllChem

from CAT.jobs import job_single_point, job_geometry_opt
from CAT.utils import restart_init
from CAT.logger import logger
from CAT.mol_utils import round_coords
from CAT.settings_dataframe import SettingsDataFrame


__all__ = ['init_asa']

# Aliases for pd.MultiIndex columns
MOL: Tuple[str, str] = ('mol', '')
ASA_INT: Tuple[str, str] = ('ASA', 'E_int')
ASA_STRAIN: Tuple[str, str] = ('ASA', 'E_strain')
ASA_E: Tuple[str, str] = ('ASA', 'E')
SETTINGS1: Tuple[str, str] = ('settings', 'ASA 1')
JOB_SETTINGS_ASA: Tuple[str, str] = ('job_settings_ASA', '')


def init_asa(qd_df: SettingsDataFrame) -> None:
    """Initialize the activation-strain analyses (ASA).

    The ASA (RDKit UFF level) is conducted on the ligands in the absence of the core.

    Parameters
    ----------
    |CAT.SettingsDataFrame|_
        A dataframe of quantum dots.

    """
    # Unpack arguments
    db = qd_df.settings.optional.database.db
    write = db and 'qd' in qd_df.settings.database.write

    # Extract any (optional) custom job types and settings
    settings = qd_df.settings.optional.qd
    asa = settings.asa
    job_recipe = {'job': asa.job1, 's': asa.s1} if isinstance(settings.asa, dict) else None
    keep_files = True if not job_recipe else settings.asa.keep_files
    path = settings.dirname

    # Prepare columns
    columns = [ASA_INT, ASA_STRAIN, ASA_E]
    for i in columns:
        qd_df[i] = np.nan

    # Fill columns
    if job_recipe is not None:
        restart_init(path, folder='asa')
    qd_df[columns] = get_asa_energy(qd_df[MOL], job_recipe)
    if job_recipe is not None:
        qd_df[JOB_SETTINGS_ASA] = get_job_settings(qd_df)
        finish()

    # Export results
    if write:
        to_db(qd_df)

    # Delte files
    if not keep_files:
        rmtree(join(path, 'asa'))


def get_job_settings(qd_df: SettingsDataFrame) -> List[str]:
    """Create a nested list of input files for each molecule in **ligand_df**."""
    job_settings = []
    for mol in qd_df[MOL]:
        try:
            job_settings.append(mol.properties.pop('job_path'))
        except KeyError:
            job_settings.append([])
    return job_settings


def to_db(qd_df: SettingsDataFrame, job_recipe: Optional[Settings] = None) -> None:
    """Export the ASA results to the database."""
    settings = qd_df.settings.optional
    db = settings.optional.database.db
    overwrite = 'qd' in settings.database.overwrite

    # Construct a job recipe
    if job_recipe is None:
        recipe = Settings({'ASA 1': {'key': 'RDKit_' + rdkit.__version__, 'value': 'UFF'}})
        columns = [SETTINGS1, ASA_INT, ASA_STRAIN, ASA_E]
    else:
        recipe = Settings({'ASA 1': {'key': job_recipe.job, 'value': job_recipe.s}})
        columns = [SETTINGS1, JOB_SETTINGS_ASA, ASA_INT, ASA_STRAIN, ASA_E]

    # Update the database
    db.update_csv(
        qd_df,
        columns=columns,
        job_recipe=recipe,
        database='QD',
        overwrite=overwrite
    )


UFF = AllChem.UFFGetMoleculeForceField


def get_asa_energy(mol_list: Iterable[Molecule], job_recipe: Optional[dict] = None) -> np.ndarray:
    """Perform an activation strain analyses (ASA).

    The ASA calculates the interaction, strain and total energy.
    The ASA is performed on all ligands in the absence of the core at the UFF level (RDKit).

    Parameters
    ----------
    mol_list : :data:`Iterable<typing.Iterable>` [:class:`Molecule`]
        An iterable consisting of PLAMS molecules.

    Returns
    -------
    :math:`n*3` |np.ndarray|_ [|np.float64|_]
        An array containing E_int, E_strain and E for all *n* molecules in **mol_series**.

    """
    if job_recipe is None:
        asa_func = _asa_uff
        job_recipe = {}
    else:
        asa_func = _asa_plams

    # Perform the activation strain analyses
    ret = []
    logger.info(f'Starting activation strain analysis')
    for i, qd in enumerate(mol_list):
        mol_complete, mol_fragments = _get_asa_fragments(qd)
        ret += asa_func(mol_complete, mol_fragments, **job_recipe)
    logger.info(f'Finishing activation strain analysis')

    # Post-process and return
    ret = np.array(ret, dtype=float)
    ret.shape = -1, 4
    ret[:, 0] -= ret[:, 1]
    ret[:, 1] -= ret[:, 2] * ret[:, 3]
    ret[:, 2] = ret[:, 0] + ret[:, 1]

    # E_int, E_strain & E
    return ret[:, 0:3]


def _get_asa_fragments(qd: Molecule) -> Tuple[Molecule, List[Molecule]]:
    """Construct the fragments for an activation strain analyses.

    Parameters
    ----------
    qd : |plams.Molecule|
        A Molecule whose atoms' properties should be marked with `pdb_info.ResidueName`.
        Atoms in the core should herein be marked with ``"COR"``.

    Returns
    -------
    |plams.Molecule| and :class:`list` [|plams.Molecule|]
        A Molecule with all core atoms removed and a list of molecules,
        one for each fragment within the molecule.
        Fragments are defined based on connectivity patterns (or lack thereof).

    """
    mol_complete = qd.copy()
    core_atoms = [at for at in mol_complete if at.properties.pdb_info.ResidueName == 'COR']
    for atom in core_atoms:
        mol_complete.delete_atom(atom)

    mol_fragments = mol_complete.separate()
    return mol_complete, mol_fragments


Mol = Union[Molecule, AllChem.Mol]


def _asa_uff(mol_complete: Mol, mol_fragments: Iterable[Mol],
             **kwargs: Any) -> Tuple[float, float, float, int]:
    r"""Perform an activation strain analyses using RDKit UFF.

    Parameters
    ----------
    mol_complete : |plams.Molecule|
        A Molecule representing the (unfragmented) relaxed structure of the system of interest.

    mol_fragments : :data:`Iterable<typing.Iterable>` [|plams.Molecule|]
        An iterable of Molecules represnting the induvidual moleculair or atomic fragments
        within **mol_complete**.

    /**kwargs : :data:`Any<typing.Any>`
        Used for retaining compatbility with the signature of :func:`._asa_plams`.

    Returns
    -------
    :class:`float`, :class:`float`, :class:`float` and :class:`int`
        The energy of **mol_complete**,
        the energy of **mol_fragments**,
        the energy of an optimized fragment within **mol_fragments** and
        the total number of fragments within **mol_fragments**.

    """
    # Create RDKit molecules
    mol_complete = molkit.to_rdmol(mol_complete)
    mol_fragments = (molkit.to_rdmol(mol) for mol in mol_fragments)

    # Calculate the energy of the total system
    E_complete = UFF(mol_complete, ignoreInterfragInteractions=False).CalcEnergy()

    # Calculate the (summed) energy of each individual fragment in the total system
    E_fragments = 0.0
    for frag_count, rdmol in enumerate(mol_fragments, 1):
        E_fragments += UFF(rdmol, ignoreInterfragInteractions=False).CalcEnergy()

    # Calculate the energy of an optimizes fragment
    UFF(rdmol, ignoreInterfragInteractions=False).Minimize()
    E_fragment_opt = UFF(rdmol, ignoreInterfragInteractions=False).CalcEnergy()

    return E_complete, E_fragments, E_fragment_opt, frag_count


def _asa_plams(mol_complete: Molecule, mol_fragments: Iterable[Mol],
               job: Type[Job], s: Settings) -> Tuple[float, float, float, int]:
    """Perform an activation strain analyses with custom Job and Settings.

    Parameters
    ----------
    mol_complete : |plams.Molecule|
        A Molecule representing the (unfragmented) relaxed structure of the system of interest.

    mol_fragments : :data:`Iterable<typing.Iterable>` [|plams.Molecule|]
        An iterable of Molecules represnting the induvidual moleculair or
        atomic fragments within **mol_complete**.

    job : :data:`Type<typing.Type>` [|plams.Job|]
        The Job type for the ASA calculations.

    s : |plams.Settings|
        The Job Settings for the ASA calculations.

    Returns
    -------
    :class:`float`, :class:`float`, :class:`float` and :class:`int`
        The energy of **mol_complete**,
        the energy of **mol_fragments**,
        the energy of an optimized fragment within **mol_fragments** and
        the total number of fragments within **mol_fragments**.

    """
    # Calculate the energy of the total system
    mol_complete.round_coords()
    mol_complete.job_single_point(job, s)
    E_complete = mol_complete.properties.energy.E

    # Calculate the (summed) energy of each individual fragment in the total system
    E_fragments = 0.0
    for frag_count, mol in enumerate(mol_fragments, 1):
        mol.round_coords()
        mol.job_single_point(job, s)
        E_fragments += mol.properties.energy.E

    # Calculate the energy of an optimizes fragment
    mol.job_geometry_opt(job, s)
    E_fragment_opt = mol.properties.energy.E

    return E_complete, E_fragments, E_fragment_opt, frag_count
