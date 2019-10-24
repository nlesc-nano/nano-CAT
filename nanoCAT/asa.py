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
from itertools import cycle, chain
from collections import abc

import numpy as np

from scm.plams import Settings, Molecule, Atom
from scm.plams.core.basejob import Job
import scm.plams.interfaces.molecule.rdkit as molkit

from rdkit.Chem import AllChem

from CAT.jobs import job_single_point, job_geometry_opt
from CAT.mol_utils import round_coords
from CAT.workflows.workflow import WorkFlow
from CAT.settings_dataframe import SettingsDataFrame
from CAT.attachment.qd_opt_ff import qd_opt_ff

from .ff.ff_assignment import run_match_job

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
    workflow(get_asa_energy, qd_df, index=idx)

    # Prepare for results exporting
    qd_df[JOB_SETTINGS_ASA] = workflow.pop_job_settings(qd_df[MOL])
    job_recipe = workflow.get_recipe()
    workflow.to_db(qd_df, index=idx, job_recipe=job_recipe)


def get_asa_energy(mol_list: Iterable[Molecule],
                   read_template: bool = True,
                   jobs: Tuple[Optional[Job], ...] = (None,),
                   settings: Tuple[Optional[Settings], ...] = (None,),
                   use_ff: bool = False,
                   **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Perform an activation strain analyses (ASA).

    The ASA calculates the interaction, strain and total energy.

    Parameters
    ----------
    mol_list : :class:`Iterable<collectionc.abc.Iterable>` [:class:`Molecule`]
        An iterable consisting of PLAMS molecules.

    jobs : :class:`tuple` [|plams.Job|], optional
        A tuple that may or may not contain |plams.Job| types.
        Will default to RDKits' implementation of UFF if ``None``.

    settings : :class:`tuple` [|plams.Settings|], optional
        A tuple that may or may not contain job |plams.Settings|.
        Will default to RDKits' implementation of UFF if ``None``.

    use_ff : :class:`bool`
        Whether or not to use a composite MATCH-based forcefield.

    \**kwargs : :data:`Any<typing.Any>`
        Further keyword arguments for ensuring signature compatiblity.

    Returns
    -------
    Three :math:`n` |np.ndarray|_ [|np.float64|_]
        A tuple of 3 arrays containing :math:`E_{int}`, :math:`E_{strain}` and :math:`E`
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
    ret = []
    get_fragments = _get_asa_fragments if not use_ff else _get_asa_ff_fragments
    for i, qd in enumerate(mol_list):
        mol_complete, mol_fragments = get_fragments(qd)
        ret += asa_func(mol_complete, mol_fragments, read_template=read_template,
                        job=job, settings=settings)

    # Cast into an array and reshape
    ret = np.array(ret, dtype=float, ndmin=2)
    ret.shape = -1, 4

    # Calclate the ASA terms
    E_int = ret[:, 0] - ret[:, 1]
    E_strain = ret[:, 1] - ret[:, 2] * ret[:, 3]
    E_tot = E_int + E_strain

    # E_int, E_strain & E
    return E_int, E_strain, E_tot


def _get_asa_fragments(qd: Molecule, use_ff: bool = False) -> Tuple[Molecule, List[Molecule]]:
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
    # Delete all atoms within the core
    mol_complete = qd.copy()
    core_atoms = [at for at in mol_complete if at.properties.pdb_info.ResidueName == 'COR']
    for atom in core_atoms:
        mol_complete.delete_atom(atom)

    # Fragment the molecule and return
    mol_fragments = mol_complete.separate()
    return mol_complete, mol_fragments


def _get_asa_ff_fragments(qd: Molecule) -> Tuple[Molecule, List[Molecule]]:
    """Construct the fragments for an activation strain analyses with a custom forcefield.

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
    # Delete all atoms within the core
    mol_complete = qd.copy()
    core_atoms = [at for at in mol_complete if at.properties.pdb_info.ResidueName == 'COR']
    for atom in core_atoms:
        mol_complete.delete_atom(atom)

    # Cap the anchor atoms with hydrogen and construct a new set of ff parameters
    atom_list = [at for at in mol_complete if at.properties.anchor]
    if atom_list[0].properties.charge == -1:  # Skip neutral atoms
        _cap_atoms(atom_list)
        mol_complete.properties.name += '_frag'

    # Fragment the molecule and return
    mol_fragments = mol_complete.separate()

    # Only anionic ligands are supported
    if atom_list[0].properties.charge != -1:
        return mol_complete, mol_fragments

    # Construct new forcefield parameters for the capped ligands
    mol_frag = mol_fragments[0]
    run_match_job(mol_frag, s=Settings({'input': {'forcefield': 'top_all36_cgenff'}}))

    # Apply the new parameters to all atoms in the to-be returned molecules
    iterator = zip(chain(*mol_fragments), mol_complete, cycle(mol_frag))
    for at1, at2, at3 in iterator:
        at1.properties.symbol = at2.properties.symbol = at3.properties.symbol
        at1.properties.charge_float = at2.properties.charge_float = at3.properties.charge_float

    # Set the prm parameter which points to the created .prm file
    mol_complete.properties.prm = mol_frag.properties.prm
    for mol in mol_fragments[1:]:
        mol.properties.prm = mol_frag.properties.prm


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
               read_template: bool, job: Type[Job],
               settings: Settings) -> Tuple[float, float, float, int]:
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
    s = settings

    # Calculate the energy of the total system
    mol_complete.round_coords()
    mol_complete.properties.name += '_frags'
    mol_complete.job_single_point(job, s)
    E_complete = mol_complete.properties.energy.E

    # Calculate the (summed) energy of each individual fragment in the total system
    E_fragments = 0.0
    for frag_count, mol in enumerate(mol_fragments, 1):
        mol.round_coords()
        mol.properties.name += '_frag'
        mol.properties.path = mol_complete.properties.path
        mol.job_single_point(job, s)
        E_fragments += mol.properties.energy.E

    # Calculate the energy of an optimizes fragment
    mol.job_geometry_opt(job, s)

    E_fragment_opt = mol.properties.energy.E
    return E_complete, E_fragments, E_fragment_opt, frag_count


def _asa_plams_ff(mol_complete: Molecule, mol_fragments: Iterable[Mol],
                  read_template: bool, job: Type[Job],
                  settings: Settings) -> Tuple[float, float, float, int]:
    """Perform an activation strain analyses with custom Job, Settings and forcefield.

    Parameters
    ----------
    mol_complete : |plams.Molecule|
        A Molecule representing the (unfragmented) relaxed structure of the system of interest.

    mol_fragments : :data:`Iterable<typing.Iterable>` [|plams.Molecule|]
        An iterable of Molecules represnting the individual moleculair or
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
    s = settings

    # Calculate the energy of the total system
    mol_complete.round_coords()
    qd_opt_ff(mol_complete, job, s, name='ASA_sp', job_func=Molecule.job_single_point)
    E_complete = mol_complete.properties.energy.E

    # Calculate the (summed) energy of each individual fragment in the total system
    E_fragments = 0.0
    for frag_count, mol in enumerate(mol_fragments, 1):
        mol.round_coords()
        mol.properties.name = mol_complete.properties.name[:-1]
        mol.properties.path = mol_complete.properties.path
        qd_opt_ff(mol, job, s, name='ASA_sp', job_func=Molecule.job_single_point)
        E_fragments += mol.properties.energy.E

    # Calculate the energy of an optimized fragment
    s = Settings(s)
    s.input.motion.geo_opt.soft_update({
        'type': 'minimization', 'optimizer': 'LBFGS', 'max_iter': 200
    })
    s.input['global'].run_type = 'geometry_optimization'
    qd_opt_ff(mol, job, s, name='ASA_opt')

    E_fragment_opt = mol.properties.energy.E
    return E_complete, E_fragments, E_fragment_opt, frag_count


def _cap_atoms(atom_list: Iterable[Atom]) -> None:
    """Cap all supplied anchor atoms with hydrogens.

    Parameter
    ---------
    atom_list : :class:`Iterable<collections.abc.Iterable>` [|plams.Atom|]
        An iterable consisting of PLAMS atoms.
        The new capping hydrogens will be added to the passed atoms' molecule.

    """
    # Ensure atom_list is a sequence
    atom_list = tuple(atom_list) if not isinstance(atom_list, abc.Sequence) else atom_list

    # Set the charges of anchor atoms to 0 so they will be recognized by add_Hs()
    try:
        for at in atom_list:
            at.properties.charge = 0
    except AttributeError as ex:
        err = "The 'atom_list' parameter contains an object of invalid type: {repr(type(at))}"
        raise TypeError(err).with_traceback(ex.__traceback__)

    # Cap the molecule with hydrogens
    mol = at.mol
    rdmol = molkit.add_Hs(mol, forcefield='uff', return_rdmol=True)
    conf = rdmol.GetConformer()

    # Add the new (and only the new!) rdkit capping atoms to the initial plams molecule
    iterator = zip(atom_list, rdmol.GetAtoms()[-len(atom_list):])
    for atom_adjacent, rd_atom in iterator:
        coords = tuple(conf.GetAtomPosition(rd_atom.GetIdx()))
        atom_new = Atom(atnum=rd_atom.GetAtomicNum(), coords=coords)
        atom_new.properties = atom_adjacent.properties.copy()
        atom_new.properties.anchor = False
        mol.add_atom(atom_new, adjacent=atom_adjacent)
