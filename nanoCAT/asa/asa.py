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

from typing import Optional, Iterable, Tuple, List, Any, Type
from itertools import chain

import numpy as np

from scm.plams import Settings, Molecule
from scm.plams.core.basejob import Job
import scm.plams.interfaces.molecule.rdkit as molkit

from rdkit.Chem import AllChem

from CAT.jobs import job_single_point, job_geometry_opt
from CAT.mol_utils import round_coords
from CAT.workflows.workflow import WorkFlow
from CAT.settings_dataframe import SettingsDataFrame
from CAT.attachment.qd_opt_ff import qd_opt_ff

from .md_asa import get_asa_md

__all__ = ['init_asa']

# Aliases for pd.MultiIndex columns
MOL: Tuple[str, str] = ('mol', '')
JOB_SETTINGS_ASA: Tuple[str, str] = ('job_settings_ASA', '')

UFF = AllChem.UFFGetMoleculeForceField


def init_asa(qd_df: SettingsDataFrame) -> None:
    """Initialize the activation-strain analyses (ASA).

    The ASA (RDKit UFF level) is conducted on the ligands in the absence of the core.

    Parameters
    ----------
    |CAT.SettingsDataFrame|
        A dataframe of quantum dots.

    """
    workflow = WorkFlow.from_template(qd_df, name='asa')

    # Run the activation strain workflow
    idx = workflow.from_db(qd_df)
    if workflow.md:
        workflow(get_asa_md, qd_df, index=idx)
    else:
        workflow(get_asa_energy, qd_df, index=idx)

    # Prepare for results exporting
    qd_df[JOB_SETTINGS_ASA] = workflow.pop_job_settings(qd_df[MOL])
    job_recipe = workflow.get_recipe()
    workflow.to_db(qd_df, index=idx, job_recipe=job_recipe)


def get_asa_energy(mol_list: Iterable[Molecule],
                   read_template: bool = True,
                   jobs: Tuple[Optional[Type[Job]], ...] = (None,),
                   settings: Tuple[Optional[Settings], ...] = (None,),
                   use_ff: bool = False,
                   **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    use_ff : :class:`bool`
        Whether or not to use a composite MATCH-based forcefield.

    \**kwargs : :data:`Any<typing.Any>`
        Further keyword arguments for ensuring signature compatiblity.

    Returns
    -------
    Three :math:`n*3` |np.ndarray|_ [|np.float64|_]
        A 2D array containing :math:`E_{int}`, :math:`E_{strain}` and :math:`E`
        for all *n* molecules in **mol_series**.

    """
    if jobs == (None,):
        asa_func = _asa_uff
        job = settings = None
    else:
        asa_func = _asa_plams if not use_ff else _asa_plams_ff
        job = jobs[0]
        settings = settings[0]

    # Perform the activation strain analyses
    E_intermediate = []
    for qd in mol_list:
        ligand_list, core = _get_asa_fragments(qd)
        E_intermediate += asa_func(
            qd, ligand_list, core, read_template=read_template, job=job, settings=settings
        )

    # Cast into an array and reshape into n*4
    E_intermediate = np.array(E_intermediate, dtype=float)
    E_intermediate.shape = -1, 5

    # Calclate the ASA terms
    E_int = E_intermediate[:, 0] - (E_intermediate[:, 1] + E_intermediate[:, 2])
    E_strain = E_intermediate[:, 1] - E_intermediate[:, 3] * E_intermediate[:, 4]
    E_tot = E_int + E_strain

    # E_int, E_strain & E
    ret = np.array([E_int, E_strain, E_tot]).T

    # Set all terms to np.nan if one of the calculations failed for that system
    isnan = np.isnan(ret).any(axis=1)
    ret[isnan] = np.nan
    return ret


def _get_asa_fragments(qd: Molecule) -> Tuple[List[Molecule], Molecule]:
    """Construct the fragments for an activation strain analyses.

    Parameters
    ----------
    qd : |plams.Molecule|
        A Molecule whose atoms' properties should be marked with `pdb_info.ResidueName`.
        Atoms in the core should herein be marked with ``"COR"``.

    Returns
    -------
    :class:`list` [|plams.Molecule|] and |plams.Molecule|
        A list of ligands and the core.
        Fragments are defined based on connectivity patterns (or lack thereof).

    """
    # Delete all atoms within the core
    mol_complete = qd.copy()
    core = Molecule()
    core.properties = mol_complete.properties.copy()

    core_atoms = [at for at in mol_complete if at.properties.pdb_info.ResidueName == 'COR']
    for atom in core_atoms:
        mol_complete.delete_atom(atom)
        atom.mol = core

    core.atoms = core_atoms
    mol_complete.properties.name += '_frags'
    core.properties.name += '_core'

    # Fragment the molecule
    ligand_list = mol_complete.separate()

    # Set atomic properties
    for at1, at2 in zip(chain(*ligand_list), mol_complete):
        at1.properties.symbol = at2.properties.symbol
        at1.properties.charge_float = at2.properties.charge_float
    for at1, at2 in zip(core, qd):
        at1.properties.symbol = at2.properties.symbol
        at1.properties.charge_float = at2.properties.charge_float

    # Set the prm parameter which points to the created .prm file
    name = mol_complete.properties.name[:-1]
    path = mol_complete.properties.path
    prm = mol_complete.properties.prm
    for mol in ligand_list:
        mol.properties.name = name
        mol.properties.path = path
        mol.properties.prm = prm

    return ligand_list, core


def _asa_uff(mol_complete: Molecule, ligands: Iterable[Molecule], core: Molecule,
             read_template: bool, job: Type[Job],
             settings: Settings) -> Tuple[float, float, float, float, int]:
    r"""Perform an activation strain analyses using RDKit UFF.

    Parameters
    ----------
    mol_complete : |plams.Molecule|
        A Molecule representing the (unfragmented) relaxed structure of the system of interest.

    mol_fragments : :class:`Iterable<collections.abc.Iterable>` [|plams.Molecule|]
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
    rd_ligands = (molkit.to_rdmol(mol) for mol in ligands)

    # Calculate the energy of the total system
    E_complete = UFF(mol_complete, ignoreInterfragInteractions=False).CalcEnergy()

    # Calculate the (summed) energy of each individual fragment in the total system
    E_ligands = 0.0
    E_min = np.inf
    mol_min = None
    for ligand_count, rdmol in enumerate(rd_ligands, 1):
        E = UFF(rdmol, ignoreInterfragInteractions=False).CalcEnergy()
        E_ligands += E
        if E < E_min:
            E_min, mol_min = E, rdmol

    # One of the calculations failed; better stop now
    if np.isnan(E_ligands):
        return np.nan, np.nan, np.nan, np.nan, ligand_count

    # Calculate the energy of an optimizes fragment
    UFF(mol_min, ignoreInterfragInteractions=False).Minimize()
    E_ligand_opt = UFF(mol_min, ignoreInterfragInteractions=False).CalcEnergy()

    E_core = UFF(molkit.to_rdmol(core), ignoreInterfragInteractions=False).CalcEnergy()
    return E_complete, E_ligands, E_core, E_ligand_opt, ligand_count


def _asa_plams(mol_complete: Molecule, ligands: Iterable[Molecule], core: Molecule,
               read_template: bool, job: Type[Job],
               settings: Settings) -> Tuple[float, float, float, float, int]:
    """Perform an activation strain analyses with custom Job and Settings.

    Parameters
    ----------
    mol_complete : |plams.Molecule|
        A Molecule representing the (unfragmented) relaxed structure of the system of interest.

    ligands : :class:`Iterable<collections.abc.Iterable>` [|plams.Molecule|]
        An iterable of Molecules containing all ligands in mol_complete.

    core : |plams.Molecule|, optional
        The core molecule from **mol_complete**.

    job : :class:`type` [|plams.Job|]
        The Job type for the ASA calculations.

    settings : |plams.Settings|
        The Job Settings for the ASA calculations.

    Returns
    -------
    :class:`float`, :class:`float`, :class:`float`, :class:`float` and :class:`int`
        The energy of **mol_complete**,
        the energy of **ligands**,
        the energy of **core**,
        the energy of an optimized fragment within **ligands** and
        the total number of fragments within **ligands**.

    """
    s = settings

    # Calculate the energy of the total system
    mol_complete.round_coords()
    mol_complete.properties.name += '_frags'
    mol_complete.job_single_point(job, s)
    E_complete = mol_complete.properties.energy.E

    # Calculate the (summed) energy of each individual fragment in the total system
    E_ligands = 0.0
    E_min = np.inf
    mol_min = None
    for ligand_count, mol in enumerate(ligands, 1):
        mol.round_coords()
        mol.job_single_point(job, s)
        E = mol.properties.energy.E
        E_ligands += E
        if E < E_min:
            E_min, mol_min = E, mol

    # One of the calculations failed; better stop now
    if np.isnan(E_ligands):
        return np.nan, np.nan, np.nan, np.nan, ligand_count

    # Calculate the energy of the core
    core.job_single_point(job, s)
    E_core = mol.properties.energy.E

    # Calculate the energy of an optimizes fragment
    mol_min.job_geometry_opt(job, s)
    E_ligand_opt = mol_min.properties.energy.E

    return E_complete, E_ligands, E_core, E_ligand_opt, ligand_count


def _asa_plams_ff(mol_complete: Molecule, ligands: Iterable[Molecule], core: Molecule,
                  read_template: bool, job: Type[Job],
                  settings: Settings) -> Tuple[float, float, float, float, int]:
    """Perform an activation strain analyses with custom Job, Settings and forcefield.

    Parameters
    ----------
    mol_complete : |plams.Molecule|
        A Molecule representing the (unfragmented) relaxed structure of the system of interest.

    ligands : :class:`Iterable<collections.abc.Iterable>` [|plams.Molecule|]
        An iterable of Molecules containing all ligands in mol_complete.

    core : |plams.Molecule|, optional
        The core molecule from **mol_complete**.

    job : :class:`type` [|plams.Job|]
        The Job type for the ASA calculations.

    settings : |plams.Settings|
        The Job Settings for the ASA calculations.

    Returns
    -------
    :class:`float`, :class:`float`, :class:`float`, :class:`float` and :class:`int`
        The energy of **mol_complete**,
        the energy of **ligands**,
        the energy of **core**,
        the energy of an optimized fragment within **ligands** and
        the total number of fragments within **ligands**.

    """
    s = Settings(settings)

    # Calculate the (summed) energy of each individual ligand fragment in the total system
    E_ligands = 0.0
    E_min = np.inf
    mol_min = None
    for ligand_count, mol in enumerate(ligands, 1):
        mol.round_coords()
        qd_opt_ff(mol, job, s, name='ASA_sp', new_psf=True, job_func=Molecule.job_single_point)
        E = mol.properties.energy.E
        E_ligands += E if E is not None else np.nan
        if E < E_min:
            E_min, mol_min = E, mol

    # Calculate the energy of the core fragment
    core.round_coords()
    qd_opt_ff(core, job, s, name='ASA_sp', new_psf=True, job_func=Molecule.job_single_point)
    E_core = core.properties.energy.E

    # Calculate the energy of the total system
    mol_complete.round_coords()
    qd_opt_ff(mol_complete, job, s, name='ASA_sp', new_psf=True, job_func=Molecule.job_single_point)
    E_complete = mol_complete.properties.energy.E

    # One of the calculations failed; better stop now
    if np.isnan(E_ligands):
        return np.nan, np.nan, np.nan, np.nan, ligand_count

    # Calculate the energy of an optimized fragment
    s.input.motion.geo_opt.soft_update({
        'type': 'minimization',
        'optimizer': 'LBFGS',
        'max_iter': 1000,
        'lbfgs': {'max_h_rank': 100}
    })
    s.input['global'].run_type = 'geometry_optimization'
    qd_opt_ff(mol_min, job, s, name='ASA_opt')
    E_ligand_opt = mol_min.properties.energy.E

    return E_complete, E_ligands, E_core, E_ligand_opt, ligand_count
