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

from rdkit import Chem
from scm.plams import Settings, Molecule, Cp2kJob, Units
from scm.plams.core.basejob import Job
import scm.plams.interfaces.molecule.rdkit as molkit

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
        * The energy of a single optimized ligand
        * The number of ligands

    """
    for mol in mol_list:
        # Run the MD job
        results = qd_opt_ff(mol, job, settings, name='QD_MD', job_func=Molecule.job_md)
        if results.job.status == 'crashed':
            yield np.nan, np.nan, np.nan, np.nan, 0
            continue

        multi_mol = MultiMolecule.from_xyz(results['cp2k-pos-1.xyz'])
        psf = PSFContainer.read(results['QD_MD.psf'])

        # Switch to the neutral parameters
        frags, _ = get_asa_fragments(mol)
        frag = frags[0]
        frag.round_coords()
        frag_count = len(frags)
        frag_neutral = frag.copy()
        frag_neutral[len(mol) - mol.properties.indices[-1] - 1].properties.charge = 0
        frag_neutral = add_Hs(frag_neutral, forcefield='uff')
        run_match_job(frag_neutral, MATCH_SETTINGS)

        # Update the PSFContainer
        psf_neutral = psf.copy()
        symbol_list = [at.properties.symbol for at in frag_neutral.atoms[:-1]] * frag_count
        charge_list = [at.properties.charge_float for at in frag_neutral.atoms[:-1]] * frag_count
        psf_neutral.atom_type.loc[psf_neutral.residue_name == 'LIG'] = symbol_list
        psf_neutral.charge.loc[psf_neutral.residue_name == 'LIG'] = charge_list

        psf_neutral.charge.loc[mol.properties.indices] += frag_neutral[-1].properties.charge_float
        psf_neutral.charge.loc[psf_neutral.residue_name == 'COR'] = 0.0

        # Calculate all inter-ligand interactions
        prm_neutral = PRMContainer.read(frag_neutral.properties.prm)
        inter_nb = _inter_nonbonded(multi_mol, None, psf_neutral, prm_neutral)

        # Calculate all intra-ligand interactions
        prm_charge = PRMContainer.read(results['QD_MD.prm'])
        intra_nb = _intra_nonbonded(multi_mol, psf, prm_charge)
        inter_bond = _inter_bonded(multi_mol, psf, prm_charge)

        # Optimize an (individual) ligand
        results = qd_opt_ff(frag, job, md2opt(settings), new_psf=True, name='Ligand_opt')

        # Calculate the optimized ligand energy
        frag_multi = MultiMolecule.from_Molecule(frag)
        psf_lig = join(results.job.path, 'Ligand_opt.psf')
        frag_opt = _intra_nonbonded(frag_multi, psf_lig, prm_charge)
        frag_opt += _inter_bonded(frag_multi, psf_lig, prm_charge)

        yield inter_nb, intra_nb, inter_bond, frag_opt, frag_count


def md2opt(s: Settings) -> Settings:
    """Convert CP2K MD settings to CP2K geometry optimization settings."""
    s2 = s.copy()
    del s2.input.motion.md
    s2.input['global'].run_type = 'geometry_optimization'

    # Delete all user-specified parameters; rely on MATCH
    del s2.input.force_eval.mm.forcefield.charge
    del s2.input.force_eval.mm.forcefield.nonbonded
    return s2


def _inter_nonbonded(multi_mol: MultiMolecule, s: Settings,
                     psf: PSFContainer, prm: PRMContainer) -> float:
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


# Temporary replacement for molkit's add_Hs() function until the function is fixed
# https://github.com/SCM-NV/PLAMS/pull/77

def add_Hs(mol, forcefield=None, return_rdmol=False):
    """Add hydrogens to protein molecules read from PDB.

    Makes sure that the hydrogens get the correct PDBResidue info.

    :param mol: Molecule to be protonated
    :type mol: |Molecule| or rdkit.Chem.Mol
    :param str forcefield: Specify 'uff' or 'mmff' to apply forcefield based
        geometry optimization on new atoms.
    :param bool return_rdmol: return a RDKit molecule if true, otherwise a PLAMS molecule
    :return: A molecule with explicit hydrogens added
    :rtype: |Molecule| or rdkit.Chem.Mol
    """
    mol = molkit.to_rdmol(mol)
    retmol = Chem.AddHs(mol)
    for atom in retmol.GetAtoms():
        if atom.GetPDBResidueInfo() is None and atom.GetSymbol() == 'H':
            bond = atom.GetBonds()[0]
            if bond.GetBeginAtom().GetIdx() == atom.GetIdx:
                connected_atom = bond.GetEndAtom()
            else:
                connected_atom = bond.GetBeginAtom()
            try:
                ResInfo = connected_atom.GetPDBResidueInfo()
                if ResInfo is None:
                    continue  # Segmentation faults are raised if ResInfo is None
                atom.SetMonomerInfo(ResInfo)
                atomname = 'H' + atom.GetPDBResidueInfo().GetName()[1:]
                atom.GetPDBResidueInfo().SetName(atomname)
            except Exception:
                pass

    unchanged = molkit.gen_coords_rdmol(retmol)
    if forcefield:
        molkit.optimize_coordinates(retmol, forcefield, fixed=unchanged)
    return retmol if return_rdmol else molkit.from_rdmol(retmol)
