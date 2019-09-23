"""
nanoCAT.ff.mol_topology
=======================

A module for constructing arrays of bond angles and proper & improper dihedral angles.

See also the :class:`.PSF` class.

Index
-----
.. currentmodule:: nanoCAT.ff.mol_topology
.. autosummary::
    get_bonds
    get_angles
    get_dihedrals
    get_impropers

API
---
.. autofunction:: nanoCAT.ff.mol_topology.get_bonds
.. autofunction:: nanoCAT.ff.mol_topology.get_angles
.. autofunction:: nanoCAT.ff.mol_topology.get_dihedrals
.. autofunction:: nanoCAT.ff.mol_topology.get_impropers

"""

import numpy as np

from scm.plams import Molecule

__all__ = ['get_bonds', 'get_angles', 'get_dihedrals', 'get_impropers']


def get_bonds(mol: Molecule) -> np.ndarray:
    """Return an array with the atomic indices defining all bonds in **mol**.

    Parameters
    ----------
    mol : |plams.Molecule|_
        A PLAMS molecule.

    Returns
    -------
    :math:`n*2` |np.ndarray|_ [|np.int64|_]:
        A 2D array with atomic indices defining :math:`n` bonds.

    """
    mol.set_atoms_id()
    bonds = [(b.atom1.id, b.atom2.id) for b in mol.bonds]

    ret = np.array(bonds, dtype=int, ndmin=2)
    mol.unset_atoms_id()
    if not bonds:  # If no angles are found
        return ret

    # Sort horizontally
    idx1 = np.argsort(ret, axis=1)
    ret[:] = np.take_along_axis(ret, idx1, axis=1)

    # Sort and return vertically
    idx2 = np.argsort(ret, axis=0)[:, 0]
    return ret[idx2]


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
    mol.set_atoms_id()
    angle = []

    for at2 in mol.atoms:
        if len(at2.bonds) < 2:
            continue

        at_other = [bond.other_end(at2) for bond in at2.bonds]
        for i, at1 in enumerate(at_other, 1):
            for at3 in at_other[i:]:
                angle.append((at1.id, at2.id, at3.id))

    ret = np.array(angle, dtype=int, ndmin=2)
    mol.unset_atoms_id()
    if not angle:  # If no angles are found
        return ret

    # Sort horizontally
    idx1 = np.argsort(ret[:, [0, -1]], axis=1)
    ret[:, 0], ret[:, -1] = np.take_along_axis(ret[:, [0, -1]], idx1, axis=1).T

    # Sort and return vertically
    idx2 = np.argsort(ret, axis=0)[:, 0]
    return ret[idx2]


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
    mol.set_atoms_id()
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

    ret = np.array(dihed, dtype=int, ndmin=2)
    mol.unset_atoms_id()
    if not dihed:  # If no dihedrals are found
        return ret

    # Sort horizontally
    idx1 = np.argsort(ret[:, [0, -1]], axis=1)
    ret[:, 0], ret[:, -1] = np.take_along_axis(ret[:, [0, -1]], idx1, axis=1).T

    # Sort and return vertically
    idx2 = np.argsort(ret, axis=0)[:, 0]
    return ret[idx2]


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
    mol.set_atoms_id()
    impropers = []

    for at1 in mol.atoms:
        order = [bond.order for bond in at1.bonds]
        if len(order) != 3:
            continue

        if 2.0 in order or 1.5 in order:
            at2, at3, at4 = [bond.other_end(at1) for bond in at1.bonds]
            impropers.append((at1.id, at2.id, at3.id, at4.id))

    ret = np.array(impropers, dtype=int, ndmin=2)
    mol.unset_atoms_id()
    if not impropers:  # If no impropers are found
        return ret

    # Sort along the rows of columns 2, 3 & 4 based on atomic mass in descending order
    mass = np.array([[mol[int(j)].mass for j in i] for i in ret[:, 1:]])
    idx = np.argsort(mass, axis=1)[:, ::-1]
    ret[:, 1], ret[:, 2], ret[:, 3] = np.take_along_axis(ret[:, 1:], idx, axis=1).T
    return ret
