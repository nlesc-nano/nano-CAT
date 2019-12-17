"""
nanoCAT.asa.asa_frag
====================

A module for creating activation strain analysis (ASA) fragments

Index
-----
.. currentmodule:: nanoCAT.asa.asa_frag
.. autosummary::
    get_asa_fragments

API
---
.. autofunction:: get_asa_fragments

"""

from typing import Tuple, List
from itertools import chain

from scm.plams import Molecule

__all__ = ['get_asa_fragments']


def get_asa_fragments(qd: Molecule) -> Tuple[List[Molecule], Molecule]:
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
