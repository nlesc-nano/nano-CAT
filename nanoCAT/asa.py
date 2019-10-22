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

from typing import Optional, Union, Iterable, Tuple, List, Any, Type

import numpy as np

from scm.plams import Settings, Molecule
from scm.plams.core.basejob import Job
import scm.plams.interfaces.molecule.rdkit as molkit

from rdkit.Chem import AllChem

from CAT.workflows.workflow import WorkFlow
from CAT.jobs import job_single_point, job_geometry_opt
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

UFF = AllChem.UFFGetMoleculeForceField


def init_asa(qd_df: SettingsDataFrame) -> None:
    """Initialize the activation-strain analyses (ASA).

    The ASA (RDKit UFF level) is conducted on the ligands in the absence of the core.

    Parameters
    ----------
    |CAT.SettingsDataFrame|_
        A dataframe of quantum dots.

    """
    workflow = WorkFlow.from_template(qd_df, name='asa')

    idx = workflow.from_db(qd_df)
    workflow(get_asa_energy, qd_df, index=idx)

    job_recipe = workflow.get_recipe()
    workflow.to_db(qd_df, index=idx, job_recipe=job_recipe)


def get_asa_energy(mol_list: Iterable[Molecule],
                   read_template: bool = True,
                   jobs: Tuple[Optional[Job], ...] = (None,),
                   settings: Tuple[Optional[Settings], ...] = (None,),
                   **kwargs: Any) -> np.ndarray:
    r"""Perform an activation strain analyses (ASA).

    The ASA calculates the interaction, strain and total energy.

    Parameters
    ----------
    mol_list : :data:`Iterable<typing.Iterable>` [:class:`Molecule`]
        An iterable consisting of PLAMS molecules.

    jobs : :class:`tuple` [|plams.Job|], optional
        A tuple that may or may not contain |plams.Job| types.
        Will default to RDKits' implementation of UFF if ``None``.

    settings : :class:`tuple` [|plams.Settings|], optional
        A tuple that may or may not contain job |plams.Settings|.
        Will default to RDKits' implementation of UFF if ``None``.

    \**kwargs : :data:`Any<typing.Any>`
        Further keyword arguments for ensuring signature compatiblity.

    Returns
    -------
    :math:`n*3` |np.ndarray|_ [|np.float64|_]
        An array containing E_int, E_strain and E for all *n* molecules in **mol_series**.

    """
    if jobs == (None,):
        asa_func = _asa_uff
        job = settings = None
    else:
        asa_func = _asa_plams
        job = jobs[0]
        settings = settings[0]

    # Perform the activation strain analyses
    ret = []
    for i, qd in enumerate(mol_list):
        mol_complete, mol_fragments = _get_asa_fragments(qd)
        ret += asa_func(mol_complete, mol_fragments, read_template=read_template,
                        job=job, settings=settings)

    # Cast into an array and reshape
    ret = np.array(ret, dtype=float, ndmin=2)
    ret.shape = -1, 4

    # Calclate the ASA terms
    ret[:, 0] -= ret[:, 1]  # E_int
    ret[:, 1] -= ret[:, 2] * ret[:, 3]  # E_train
    ret[:, 2] = ret[:, 0] + ret[:, 1]  # E_tot

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
               read_template: bool, job: Type[Job], s: Settings) -> Tuple[float, float, float, int]:
    """Perform an activation strain analyses with custom Job and Settings.

    Parameters
    ----------
    mol_complete : |plams.Molecule|
        A Molecule representing the (unfragmented) relaxed structure of the system of interest.

    mol_fragments : :data:`Iterable<typing.Iterable>` [|plams.Molecule|]
        An iterable of Molecules represnting the induvidual moleculair or
        atomic fragments within **mol_complete**.

    read_template : :class:`bool`
        Whether or not to use the QMFlows template system.

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
