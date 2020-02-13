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

from typing import Iterable, Tuple, Any, Type, Generator, Sequence
from itertools import chain

import numpy as np

from scm.plams import Settings, Molecule, Cp2kJob, Units
from scm.plams.core.basejob import Job
from scm.plams.interfaces.molecule.rdkit import add_Hs

from FOX import MultiMolecule, PSFContainer, PRMContainer, get_intra_non_bonded, get_bonded

from CAT.jobs import job_md
from CAT.mol_utils import round_coords
from CAT.attachment.qd_opt_ff import qd_opt_ff, get_psf

from .asa_frag import get_asa_fragments
from .energy_gatherer import EnergyGatherer
from ..ff.ff_assignment import run_match_job


def get_asa_md(mol_list: Iterable[Molecule], jobs: Tuple[Type[Job], ...],
               settings: Tuple[Settings, ...], iter_start: int = 500,
               scale_elstat: float = 0.0, scale_lj: float = 1.0,
               distance_upper_bound: float = np.inf, k: int = 20,
               dump_csv: bool = False, **kwargs: Any) -> np.ndarray:
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

    iter_start : :class:`int`
        The MD iteration at which the ASA will be started.
        All preceding iteration are disgarded, treated as pre-equilibration steps.

    scale_elstat : :class:`float`
        Scaling factor to apply to all 1,4-nonbonded electrostatic interactions.
        Serves the same purpose as the cp2k ``EI_SCALE14`` keyword.

    scale_lj : :class:`float`
        Scaling factor to apply to all 1,4-nonbonded Lennard-Jones interactions.
        Serves the same purpose as the cp2k ``VDW_SCALE14`` keyword.

    distance_upper_bound : :class:`float`
        Consider only atom-pairs within this distance for calculating inter-ligand interactions.
        Units are in Angstrom.
        Using ``inf`` will default to the full, untruncated, distance matrix.

    k : :class:`int`
        The (maximum) number of to-be considered distances per atom.
        Only relevant when **distance_upper_bound** is not set to ``inf``.

    dump_csv : :class:`str`, optional
        If ``True``, dump the raw energy terms to a set of .csv files.

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
    iterator = chain.from_iterable(md_generator(mol_list, job, s,
                                                iter_start=iter_start,
                                                scale_elstat=scale_elstat,
                                                scale_lj=scale_lj,
                                                distance_upper_bound=distance_upper_bound,
                                                k=k, dump_csv=dump_csv))

    E = np.fromiter(iterator, count=count, dtype=float)
    E.shape = shape
    E[:, :4] *= Units.conversion_ratio('hartree', 'kcal/mol')

    # Calculate (and return) the interaction, strain and total energy
    E_int = E[:, 0]
    E_strain = E[:, 1:3].sum(axis=1) - E[:, 3:].prod(axis=1)
    return np.array([E_int, E_strain, E_int + E_strain]).T


MATCH_SETTINGS = Settings({'input': {'forcefield': 'top_all36_cgenff'}})
Tuple5 = Tuple[float, float, float, float, int]


def md_generator(mol_list: Iterable[Molecule], job: Type[Job],
                 settings: Settings, iter_start: int = 500,
                 scale_elstat: float = 0.0, scale_lj: float = 1.0,
                 distance_upper_bound: float = np.inf,
                 k: int = 20, dump_csv: bool = False) -> Generator[Tuple5, None, None]:
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

    iter_start : :class:`int`
        The MD iteration at which the ASA will be started.
        All preceding iteration are disgarded, treated as pre-equilibration steps.

    scale_elstat : :class:`float`
        Scaling factor to apply to all 1,4-nonbonded electrostatic interactions.
        Serves the same purpose as the cp2k ``EI_SCALE14`` keyword.

    scale_lj : :class:`float`
        Scaling factor to apply to all 1,4-nonbonded Lennard-Jones interactions.
        Serves the same purpose as the cp2k ``VDW_SCALE14`` keyword.

    distance_upper_bound : :class:`float`
        Consider only atom-pairs within this distance for calculating inter-ligand interactions.
        Units are in Angstrom.
        Using ``inf`` will default to the full, untruncated, distance matrix.

    k : :class:`int`
        The (maximum) number of to-be considered distances per atom.
        Only relevant when **distance_upper_bound** is not set to ``inf``.

    dump_csv : :class:`str`, optional
        If ``True``, dump the raw energy terms to a set of .csv files.

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
        prm_charged = PRMContainer.read(mol.properties.prm)
        psf_lig = get_psf(ligands[0], charges=None)

        # import pdb; pdb.set_trace()
        lig = _get_best_ligand(ligands, psf_lig, prm_charged)
        lig.round_coords()
        lig_count = len(ligands)

        # Run the MD job
        md_results = qd_opt_ff(mol, job, settings, name='QD_MD', job_func=Molecule.job_md)
        if md_results.job.status == 'crashed':
            yield np.nan, np.nan, np.nan, np.nan, 0
            continue

        md_trajec = MultiMolecule.from_xyz(md_results['cp2k-pos-1.xyz'])[iter_start:]
        psf_charged = PSFContainer.read(md_results['QD_MD.psf'])

        # Optimize a single ligand
        opt_results = qd_opt_ff(lig, job, _md2opt(settings), new_psf=True, name='ligand_opt')
        if opt_results.job.status == 'crashed':
            yield np.nan, np.nan, np.nan, np.nan, 0
            continue

        # Prepare arguments for the intra-ligand interactions
        lig_opt = MultiMolecule.from_Molecule(lig)

        # Prepare arguments for the inter-ligand interactions
        lig_neutral = _get_neutral_frag(lig)
        prm_neutral = PRMContainer.read(lig_neutral.properties.prm)
        psf_neutral = _get_neutral_psf(psf_charged, lig_neutral,
                                       lig_count, mol.properties.indices)

        # Inter-ligand interaction
        qd_map = EnergyGatherer()
        inter_nb = qd_map.inter_nonbonded(md_trajec, None, psf_neutral, prm_neutral,
                                          distance_upper_bound=distance_upper_bound, k=k)

        # Intra-ligand interaction
        psf_charged.charge = [(at.properties.charge_float if at.properties.charge_float else 0.0) for at in mol]
        intra_bond = qd_map.intra_bonded(md_trajec, psf_charged, prm_charged)
        intra_nb = qd_map.intra_nonbonded(md_trajec, psf_charged, prm_charged,
                                          scale_elstat=scale_elstat, scale_lj=scale_lj)

        # Intra-ligand interaction within a single optimized ligand
        lig_map = EnergyGatherer()
        frag_opt = lig_map.intra_bonded(lig_opt, psf_lig, prm_charged)
        frag_opt += lig_map.intra_nonbonded(lig_opt, psf_lig, prm_charged,
                                            scale_elstat=scale_elstat, scale_lj=scale_lj)

        import pandas as pd

        b = {'angles', 'bonds', 'dihedrals', 'impropers'}

        strain = pd.Series(0.0, index=qd_map.angles.index.copy())
        for k, v in qd_map.items():
            if k in {'inter_elstat', 'inter_lj'}:
                continue
            if k in b:
                continue
            elif v is None:
                continue
            strain += np.nansum(v, axis=1)

        strain /= lig_count

        for k, v in lig_map.items():
            if v is None:
                continue
            if k in b:
                continue
            strain -= np.nansum(v, axis=1)

        strain *= Units.conversion_ratio('au', 'kcal/mol')

        E_bond = qd_map.bonds.sum(axis=1) / lig_count - lig_map.bonds.sum(axis=1).values
        E_angles = qd_map.angles.sum(axis=1) / lig_count - lig_map.angles.sum(axis=1).values
        E_impropers = qd_map.impropers.sum(axis=1) / lig_count - lig_map.impropers.sum(axis=1).values
        E_dihedrals = qd_map.dihedrals.sum(axis=1) / lig_count - lig_map.dihedrals.sum(axis=1).values
        E_lj = qd_map.intra_lj.sum(axis=1) / lig_count - lig_map.intra_lj.sum(axis=1).values
        E_elstat = qd_map.intra_elstat.sum(axis=1) / lig_count - lig_map.intra_elstat.sum(axis=1).values

        E_bond *= Units.conversion_ratio('au', 'kcal/mol')
        E_angles *= Units.conversion_ratio('au', 'kcal/mol')
        E_impropers *= Units.conversion_ratio('au', 'kcal/mol')
        E_dihedrals *= Units.conversion_ratio('au', 'kcal/mol')
        E_lj *= Units.conversion_ratio('au', 'kcal/mol')
        E_elstat *= Units.conversion_ratio('au', 'kcal/mol')

        yield inter_nb, intra_nb, intra_bond, frag_opt, lig_count


def _get_best_ligand(ligand_list: Sequence[Molecule], psf, prm) -> Molecule:
    mol = MultiMolecule.from_Molecule(ligand_list)

    elstat, lj = get_intra_non_bonded(mol, psf, prm)
    series = elstat.sum(axis=1) + lj.sum(axis=1)
    for i in get_bonded(mol, psf, prm):
        if i is not None:
            series += i.sum(axis=1)

    i = series.values.argmin()
    return ligand_list[i]


def _get_neutral_frag(frag: Molecule) -> Molecule:
    """Return a neutral fragment for :func:`md_generator`."""
    frag_neutral = frag.copy()
    for at in frag_neutral:
        if at.properties.anchor:
            at.properties.charge = 0
            break

    frag_neutral = add_Hs(frag_neutral, forcefield='uff')
    frag_neutral.properties.pdb_info.IsHeteroAtom = False
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
