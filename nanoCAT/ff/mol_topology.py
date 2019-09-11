"""
nanoCAT.ff.mol_topology
=======================

A module for constructing arrays of bond angles and proper & improper dihedral angles.

Index
-----
.. currentmodule:: nanoCAT.ff.mol_topology
.. autosummary::
    get_angles
    get_dihedrals
    get_impropers

API
---
.. autofunction:: nanoCAT.ff.mol_topology
    get_angles
    get_dihedrals
    get_impropers

"""

import numpy as np

from scm.plams import Molecule

__all__ = ['get_angles', 'get_dihedrals', 'get_impropers']


def get_angles(mol: Molecule) -> np.ndarray:
    """Return an array with the atomic indices defining all angles in **mol**.

    Parameters
    ----------
    mol : |plams.Molecule|_
        A PLAMS molecule.

    Returns
    -------
    :math:`n*3` |np.ndarray|_ [|np.int64|_]:
        A 2D array with atomic indices defining :math:`n` angles.

    """
    mol.set_atoms_id(start=0)
    angle = []

    for at2 in mol.atoms:
        if len(at2.bonds) < 2:
            continue

        at_other = [bond.other_end(at2) for bond in at2.bonds]
        for i, at1 in enumerate(at_other, 1):
            for at3 in at_other[i:]:
                angle.append((at1.id, at2.id, at3.id))

    ret = np.array(angle, dtype=int, ndmin=2) + 1
    if not angle:  # If no angles are found
        return ret

    # Sort horizontally
    for i, (j, k, m) in enumerate(ret):
        if j > m:
            ret[i] = (m, k, j)

    # Sort and return vertically
    idx = np.argsort(ret, axis=0)[:, 0]
    return ret[idx]


def get_dihedrals(mol: Molecule) -> np.ndarray:
    """Return an array with the atomic indices defining all proper dihedral angles in **mol**.

    Parameters
    ----------
    mol : |plams.Molecule|_
        A PLAMS molecule.

    Returns
    -------
    :math:`n*4` |np.ndarray|_ [|np.int64|_]:
        A 2D array with atomic indices defining :math:`n` proper dihedrals.

    """
    mol.set_atoms_id(start=0)
    dihed = []

    for b1 in mol.bonds:
        if not (len(b1.atom1.bonds) > 1 and len(b1.atom2.bonds) > 1):
            continue

        at2, at3 = b1
        for b2 in at2.bonds:
            at1 = b2.other_end(at2)
            if at1 == at3:
                continue

            for b3 in at3.bonds:
                at4 = b3.other_end(at3)
                if at4 != at2:
                    dihed.append((at1.id, at2.id, at3.id, at4.id))

    ret = np.array(dihed, dtype=int, ndmin=2) + 1
    if not dihed:  # If no dihedrals are found
        return ret

    # Sort horizontally
    for i, (j, k, m, n) in enumerate(ret):
        if j > n:
            ret[i] = (n, m, k, j)

    # Sort and return vertically
    idx = np.argsort(ret, axis=0)[:, 0]
    return ret[idx]


def get_impropers(mol: Molecule) -> np.ndarray:
    """Return an array with the atomic indices defining all improper dihedral angles in **mol**.

    Parameters
    ----------
    mol : |plams.Molecule|_
        A PLAMS molecule.

    Returns
    -------
    :math:`n*4` |np.ndarray|_ [|np.int64|_]:
        A 2D array with atomic indices defining :math:`n` improper dihedrals.

    """
    mol.set_atoms_id(start=0)
    impropers = []

    for at1 in mol.atoms:
        order = [bond.order for bond in at1.bonds]
        if len(order) != 3:
            continue

        if 2.0 in order or 1.5 in order:
            at2, at3, at4 = [bond.other_end(at1) for bond in at1.bonds]
            impropers.append((at1.id, at2.id, at3.id, at4.id))

    ret = np.array(impropers, dtype=int, ndmin=2) + 1
    if not impropers:  # If no impropers are found
        return ret

    # Sort along the rows of columns 2, 3 & 4 based on atomic mass in descending order
    mass = np.array([[mol[j].mass for j in i] for i in ret[:, 1:]])
    idx = np.argsort(mass, axis=1)[:, ::-1]
    for i, j in enumerate(idx):
        ret[i, 1:] = ret[i, 1:][j]

    return ret
