"""
nanoCAT.ff.ff_cationic
======================

A worklflow for assigning CHARMM forcefield parameters to molecules.

Index
-----
.. currentmodule:: nanoCAT.ff.ff_cationic
.. autosummary::
    get_ff_cationic

API
---
.. autofunction:: get_ff_cationic

"""

from scm.plams import Molecule, Atom, Settings

from CAT.attachment.mol_split_cm import SplitMol
from nanoCAT.ff.ff_assignment import run_match_job

__all__ = ['get_ff_cationic']


def get_ff_cationic(mol: Molecule, anchor: Atom,
                    forcefield: str = 'top_all36_cgenff') -> Molecule:
    s = Settings({'input': {'forcefield': forcefield}})

    bonds = anchor.bonds.copy()
    for b in bonds:
        with SplitMol(mol, b) as (mol1, mol2):
            pass
