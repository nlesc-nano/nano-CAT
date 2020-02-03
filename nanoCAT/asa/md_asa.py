"""
nanoCAT.asa.md_asa
==================

A module related to performing MD-averaged activation strain analyses.

Index
-----
.. currentmodule:: nanoCAT.asa.md_asa
.. autosummary::
    get_asa_md
    md_generator

API
---
.. autofunction:: get_asa_md
.. autofunction:: md_generator

"""

from os.path import join
from typing import Iterable, Tuple, Any, Type, Generator
from itertools import chain

import numpy as np
import pandas as pd

from scm.plams import Settings, Molecule, Cp2kJob, Units
from scm.plams.core.basejob import Job
from scm.plams.interfaces.molecule.rdkit import add_Hs

from FOX import (
    get_non_bonded, get_intra_non_bonded, get_bonded, MultiMolecule, PSFContainer, PRMContainer
)

from CAT.jobs import job_md
from CAT.mol_utils import round_coords
from CAT.attachment.qd_opt_ff import qd_opt_ff

from .asa_frag import get_asa_fragments
from ..ff.ff_assignment import run_match_job


def get_asa_md(mol_list: Iterable[Molecule], jobs: Tuple[Type[Job], ...],
               settings: Tuple[Settings, ...], **kwargs: Any) -> np.ndarray:
    r"""Perform an activation strain analyses (ASA) along an molecular dynamics (MD) trajectory.

    The ASA calculates the (ensemble-averaged) interaction, strain and total energy.

    Parameters
    ----------
    mol_list : :class:`Iterable<collectionc.abc.Iterable>` [:class:`Molecule`]
        An iterable consisting of PLAMS molecules.

    jobs : :class:`tuple` [|plams.Job|]
        A tuple containing a single |plams.Job| type.

    settings : :class:`tuple` [|plams.Settings|]
        A tuple containing a single |plams.Settings| instance.

    \**kwargs : :data:`Any<typing.Any>`
        Further keyword arguments for ensuring signature compatiblity.

    Returns
    -------
    :math:`n*3` |np.ndarray|_ [|np.float64|_]
        Returns a 2D array respectively containing :math:`E_{int}`, :math:`E_{strain}`
        and :math:`E`.
        Ensemble-averaged energies are calculated for the, to-be computed, MD trajectories of
        all *n* molecules in **mol_list**.

    """
    # Extract all Job types and job Settings
    job = jobs[0]
    s = settings[0].copy()
    if job is not Cp2kJob:
        raise ValueError("'jobs' expected '(Cp2kJob,)'; observed value: {repr(jobs)}")

    # Infer the shape of the to-be created energy array
    try:
        mol_len = len(mol_list)
    except TypeError:  # **mol_list*** is an iterator
        shape = -1, 5
        count = -1
    else:
        shape = mol_len, 5
        count = mol_len * 5

    # Extract all energies and ligand counts
    iterator = chain.from_iterable(md_generator(mol_list, job, s))
    E = np.fromiter(iterator, count=count, dtype=float)
    E.shape = shape
    E[:, :4] *= Units.conversion_ratio('au', 'kcal/mol')

    # Calculate (and return) the interaction, strain and total energy
    E_int = E[:, 0]
    E_strain = np.sum(E[:, 1:3], axis=1) - np.product(E[:, 3:], axis=1)

    ret = np.array([E_int, E_strain, E_int + E_strain]).T
    return ret


MATCH_SETTINGS = Settings({'input': {'forcefield': 'top_all36_cgenff'}})
KCAL2AU: float = Units.conversion_ratio('kcal/mol', 'hartree')  # kcal/mol to hartree
Tuple5 = Tuple[float, float, float, float, int]


def md_generator(mol_list: Iterable[Molecule], job: Type[Job],
                 settings: Settings) -> Generator[Tuple5, None, None]:
    """Iterate over an iterable of molecules; perform an MD followed by an ASA.

    The various energies are averaged over all molecules in the MD-trajectory.

    Parameters
    ----------
    mol_list : :class:`Iterable<collectionc.abc.Iterable>` [:class:`Molecule`]
        An iterable consisting of PLAMS molecules.

    job : |plams.Job|
        A |plams.Job| type.
        Should be equal to :class:`Cp2kJob`.

    settings : :class:`tuple` [|plams.Settings|]
        CP2K job settings for **job**.

    Returns
    -------
    4x :class:`float` and 1x :class:`int`
        A tuple with 5 (ensemble-averaged) quantities:

        * The inter-ligand non-bonded interaction
        * The intra-ligand non-bonded interaction
        * The intra-ligand bonded interaction
        * The energy of a single optimized ligand (bonded & non-bonded interactions)
        * The number of ligands

    """
    for mol in mol_list:
        # Identify the fragments
        ligands, _ = get_asa_fragments(mol)
        lig = ligands[0]
        lig.round_coords()
        lig_count = len(ligands)

        # Run the MD job
        md_results = qd_opt_ff(mol, job, settings, name='QD_MD', job_func=Molecule.job_md)
        if md_results.job.status == 'crashed':
            yield np.nan, np.nan, np.nan, np.nan, 0
            continue

        md_trajec = MultiMolecule.from_xyz(md_results['cp2k-pos-1.xyz'])
        psf_charged = PSFContainer.read(md_results['QD_MD.psf'])

        # Optimize a single ligand
        opt_results = qd_opt_ff(lig, job, _md2opt(settings), new_psf=True, name='ligand_opt')
        if opt_results.job.status == 'crashed':
            yield np.nan, np.nan, np.nan, np.nan, 0
            continue

        # Prepare arguments for the intra-ligand interactions
        lig_opt = MultiMolecule.from_Molecule(lig)
        prm_charged = PRMContainer.read(opt_results['ligand_opt.prm'])
        psf_lig = join(opt_results.job.path, 'ligand_opt.psf')

        # Prepare arguments for the inter-ligand interactions
        lig_neutral = _get_neutral_frag(lig)
        prm_neutral = PRMContainer.read(lig_neutral.properties.prm)
        psf_neutral = _get_neutral_psf(psf_charged, lig_neutral,
                                       lig_count, mol.properties.indices)

        # Inter-ligand interaction
        inter_nb = _inter_nonbonded(md_trajec, None, psf_neutral, prm_neutral)

        # Intra-ligand interaction
        intra_nb = _intra_nonbonded(md_trajec, psf_charged, prm_charged)
        intra_bond = _inter_bonded(md_trajec, psf_charged, prm_charged)

        # Intra-ligand interaction within a single optimized ligand
        frag_opt = _intra_nonbonded(lig_opt, psf_lig, prm_charged)
        frag_opt += _inter_bonded(lig_opt, psf_lig, prm_charged)

        yield inter_nb, intra_nb, intra_bond, frag_opt, lig_count


def _get_neutral_frag(frag: Molecule) -> Molecule:
    """Return a neutral fragment for :func:`md_generator`."""
    frag_neutral = frag.copy()
    for at in frag_neutral:
        if at.properties.anchor:
            at.properties.charge = 0
            break
    frag_neutral = add_Hs(frag_neutral, forcefield='uff')
    run_match_job(frag_neutral, MATCH_SETTINGS)
    return frag_neutral


def _get_neutral_psf(psf: PSFContainer, frag_neutral: Molecule, frag_count: int,
                     anchor_idx: np.ndarray) -> PSFContainer:
    """Return a net-neutral :class:`PSFContainer` for :func:`md_generator`."""
    psf_neutral = psf.copy()
    symbol_list = [at.properties.symbol for at in frag_neutral.atoms[:-1]] * frag_count
    charge_list = [at.properties.charge_float for at in frag_neutral.atoms[:-1]] * frag_count

    psf_neutral.atom_type.loc[psf_neutral.residue_name == 'LIG'] = symbol_list
    psf_neutral.charge.loc[psf_neutral.residue_name == 'LIG'] = charge_list

    psf_neutral.charge.loc[anchor_idx] += frag_neutral[-1].properties.charge_float
    psf_neutral.charge.loc[psf_neutral.residue_name == 'COR'] = 0.0
    return psf_neutral


def _md2opt(s: Settings) -> Settings:
    """Convert CP2K MD settings to CP2K geometry optimization settings."""
    s2 = s.copy()
    del s2.input.motion.md
    s2.input['global'].run_type = 'geometry_optimization'

    # Delete all user-specified parameters; rely on MATCH
    del s2.input.force_eval.mm.forcefield.charge
    del s2.input.force_eval.mm.forcefield.nonbonded
    return s2


def _inter_nonbonded(multi_mol: MultiMolecule, s: Settings, psf: PSFContainer,
                     prm: PRMContainer) -> float:
    """Collect all inter-ligand non-bonded interactions."""
    # Manually calculate all inter-ligand, ligand/core & core/core interactions
    df = get_non_bonded(multi_mol, psf=psf, prm=prm, cp2k_settings=s)

    # Set all core/core and core/ligand interactions to 0
    core = set(psf.atom_name[psf.residue_name == 'COR'])
    for key in df.index:
        if core.intersection(key):
            df.loc[key] = 0

    return df.values.sum()


def _intra_nonbonded(multi_mol: MultiMolecule, psf: PSFContainer, prm: PRMContainer) -> float:
    """Collect all intra-ligand non-bonded interactions."""
    return get_intra_non_bonded(multi_mol, psf=psf, prm=prm).values.sum()


def _inter_bonded(multi_mol: MultiMolecule, psf: PSFContainer, prm: PRMContainer) -> float:
    """Collect all intra-ligand bonded interactions."""
    E_tup = get_bonded(multi_mol, psf, prm)  # bonds, angles, dihedrals, impropers
    return sum((series.sum() if series is not None else 0.0) for series in E_tup)
