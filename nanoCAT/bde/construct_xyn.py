"""
nanoCAT.bde.construct_xyn
=========================

A module for constructing :math:`XYn` compounds.

Index
-----
.. currentmodule:: nanoCAT.bde.construct_xyn
.. autosummary::
    get_xyn
    get_perpendicular_vec

API
---
.. autofunction:: get_xyn
.. autofunction:: get_perpendicular_vec

"""

from __future__ import annotations

from itertools import chain
from typing import Tuple

import numpy as np
from scm.plams import MoleculeError, Molecule, Atom, AMSJob, axis_rotation_matrix

from CAT.jobs import job_geometry_opt  # noqa: F401
from CAT.utils import get_template
from CAT.mol_utils import to_atnum

__all__ = ['get_xyn', 'get_perpendicular_vec']


def get_xyn(
    mol_ref: Molecule,
    lig_count: int = 2,
    ion: None | Molecule | str | int = 'Cd',
    opt: bool = True,
) -> Molecule:
    """Combine an ion (:math:`X`) and a quantum dot with :math:`n` ligands (:math:`Y`).

    The ligands are attached to **ion** such that the resulting :math:`XY_{n}` molecule addopts
    a planar conformation (*e.g.* linear, trigonal planar or square planar).

    Examples
    --------
    Given a :math:`Cd` ion, 2 ligands and a quantum dot with acetate ligands,
    :math:`AcO^-`, the following compound would be produced
    (:code:`get_xyn(acetate_mol, lig_count=2, ion='Cd')`):

    .. math::
        Cd^{2+} + 2 * AcO^- = Cd^{2+}{AcO^-}_{2}

    Another example with polyatomic ammonium compound (:math:`{NH_4}^+`) and
    a single :math:`Cl^-` ligand (:code:`get_xyn(chloride_mol, lig_count=1, ion=NH4_mol)`):

    .. math::
        {NH_4}^+ + Cl^- = {NH_4}^+ Cl^-

    Warning
    -------
    Polyatomic ions with more than one (formally) charged atom should be provided
    at the users own risk.
    In such scenario, the first charged atom encoutered in :attr:`Molecule.atoms` will be
    treated as anchor, *i.e.* all :math:`Y_{n}` ligands will be attached to this atom.

    Parameters
    ----------
    mol_ref : |plams.Molecule|_
        A PLAMS molecule containing ligands (:math:`Y`).
        Ligands are identified within the quantum dot based on the :attr:`Atom.properties`
        ``["pdb_info"]["ResidueNumber"]`` key.

    lig_count : int
        The number of ligands (:math:`n`) which is to-be added to the ion :math:`X`.

    ion : |str|_, |int|_ or |plams.Molecule|_
        Optional: An ion (:math:`X`).
        Accepts both mono- (*i.e.* atomic number or symbol) or poly-atomic ions.
        The charge of **ion** is modified to produce a charge-neutral :math:`XY_{n}` complex.
        If ``None``, return the ligand (:math:`Y`) without an additional ion.

    opt : bool
        Perform a constrained geometry optimization (RDKit UFF) on the :math:`XY_{n}` complex.

    Returns
    -------
    |plams.Molecule|_:
        A new :math:`XY_{n}` molecule.

    """
    # Identify a ligands and its anchor atom
    lig = _get_ligand(mol_ref)
    idx, at = _get_anchor(lig)

    # Return a the ligand without an ion
    if ion is None:
        lig.properties.update({
            'name': mol_ref.properties.name + '_XYn',
            'path': mol_ref.properties.path,
            'indices': [idx],
            'job_path': []
        })
        return lig

    # Construct XYn
    XYn, X = _construct_xyn(ion, lig_count, lig, at, idx)

    # Update the properties of X and XYn
    X.properties.charge = 0 - sum([at.properties.charge for at in XYn if at.properties.charge])
    XYn.properties.update({
        'name': mol_ref.properties.name + '_XYn',
        'path': mol_ref.properties.path,
        'indices': (list(range(1, 1 + len(ion))) if isinstance(ion, Molecule) else [1]),
        'job_path': [],
        'prm': mol_ref.properties.prm
    })

    X.properties.symbol = X.symbol
    X.properties.pdb_info.ResidueName = 'COR'
    X.properties.pdb_info.ResidueNumber = 1

    # Perform a constrained UFF optimization on XYn with X frozen
    if opt:
        XYn.round_coords()
        _preoptimize(XYn)
    XYn.round_coords()

    # Update the constrains; constrain only the anchor atoms
    X_idx = 1 + XYn.atoms.index(X)
    start = idx + (len(ion) if isinstance(ion, Molecule) else 1)
    XYn.properties.indices = [X_idx] + [start + i*len(lig) for i in range(lig_count)]

    # Delete the now redundant bonds connected to X
    for bond in reversed(X.bonds):
        XYn.delete_bond(bond)

    return XYn


def _construct_xyn(
    ion: str | int | Molecule,
    lig_count: int,
    lig: Molecule,
    lig_at: Atom,
    lig_idx: int,
) -> Tuple[Molecule, Atom]:
    """Construct the :math:`XYn` molecule for :func:`get_xyn`.

    Parameters
    ----------
    ion : |str|_, |int|_ or |plams.Molecule|_
        An ion (:math:`X`), be it mono- (*e.g.* atomic number or symbol) or poly-atomic.

    lig_count : int
        The number of to-be attached ligands per ion.

    lig : |plams.Molecule|_
        A single ligand molecule.

    lig_at : |plams.Atom|_
        The ligand anchor atom.

    lig_idx : int
        The (1-based) index of **lig_at**.

    Returns
    -------
    |plams.Molecule|_ and |plams.Atom|_
        A :math:`XY_{n}` molecule and the the charged atom from :math:`X`.

    """
    # Create a list of n ligands, n anchor atoms, n desired ion-anchor distances and n angles
    lig_gen = (lig.copy() for _ in range(lig_count))
    angle_ar = np.arange(0, 2*np.pi, 2*np.pi / lig_count)

    # Prepare vectors for translations and rotations
    vec1 = lig_at.vector_to(np.zeros(3))
    _vec = lig_at.vector_to(lig.get_center_of_mass())
    vec2 = get_perpendicular_vec(_vec)

    # Update the XYn molecule with ligands
    XYn, X = _parse_ion(ion)
    iterator = enumerate(zip(angle_ar, lig_gen), 2)
    for i, (angle, mol) in iterator:
        # Prepare for translations and rotations
        anchor = mol[lig_idx]
        rotmat = axis_rotation_matrix(vec2, angle)
        dist = anchor.radius + X.radius

        # Translate and rotate the ligand
        mol.translate(vec1)
        mol.rotate(rotmat)
        vec3 = anchor.vector_to(mol.get_center_of_mass())
        vec3 /= np.linalg.norm(vec3) / dist
        mol.translate(vec3)

        # Set pdb attributes
        for at in mol:
            at.properties.pdb_info.ResidueNumber = i
            at.properties.pdb_info.ResidueName = 'LIG'

        # Combine the translated and rotated ligand with XYn
        XYn.add_molecule(mol)
        XYn.add_bond(X, anchor)

    return XYn, X


def _preoptimize(mol: Molecule) -> None:
    """Perform a constrained geometry optimization of **mol** with AMS UFF."""
    s = get_template('qd.yaml')['UFF']
    s.input.ams.constraints.atom = mol.properties.indices
    s.input.ams.GeometryOptimization.coordinatetype = 'Cartesian'
    mol.job_geometry_opt(AMSJob, s, name='E_XYn_preopt')


def _parse_ion(ion: Molecule | str | int) -> Tuple[Molecule, Atom]:
    """Interpret and parse the **ion** argument in :func:`.get_xyn`.

    Construct and return a new :math:`XY_{n=0}` molecule and the atom :math:`X` itself.
    If **ion** is a polyatomic ion then :math:`XY_{n=0}` is a copy of **ion** and :math:`X`
    is the first atom with a non-zero charge.

    Parameters
    ----------
    ion : |str|_, |int|_ or |plams.Molecule|_
        An ion (:math:`X`), be it mono- (*e.g.* atomic number or symbol) or poly-atomic.

    Returns
    -------
    |plams.Molecule|_ and |plams.Atom|_
        A :math:`XY_{n=0}` molecule and the the charged atom from :math:`X`.

    Raises
    ------
    MoleculeError
        Raised if ion is an instance of :math:`Molecule` but does not contain any charged atoms.

    """
    if isinstance(ion, Molecule):
        XYn = ion.copy()
        for i, at in enumerate(XYn, 1):
            if not at.properties.charge:
                continue

            # Found an atom with non-zero charge; return a copy
            ret = XYn.copy()
            return ret, ret[i]

        raise MoleculeError("No atoms were found in 'ion' with a non-zero charge")

    else:
        # Ion is an atomic number or symbol
        X = Atom(atnum=to_atnum(ion))
        XYn = Molecule()
        XYn.add_atom(X)
        return XYn, X


def _get_anchor(mol: Molecule) -> Tuple[int, Atom]:
    """Find the first atom in **mol** marked with :attr:`Atom.properties` ``["anchor"]``.

    Parameters
    ----------
    |plams.Molecule|_
        A PLAMS molecule containing (at least) one atom marked whose :attr:`Atom.properties`
        attribute contains the ``["anchor"]`` key.

    Returns
    -------
    |int|_ and |plams.Atom|_
        The (0-based) index and the matching Atom

    Raises
    ------
    MoleculeError
        Raised if no atom with the :attr:`Atom.properties` ``["anchor"]`` key is found.

    """
    for i, at in enumerate(mol.atoms, 1):
        if at.properties.anchor:
            return i, at

    raise MoleculeError("No atom with the Atom.properties.anchor found")


def _get_ligand(mol: Molecule) -> Molecule:
    """Extract a single ligand from **mol** as a copy."""
    at_list = []
    res = mol.atoms[-1].properties.pdb_info.ResidueNumber
    for at in reversed(mol.atoms):
        if at.properties.pdb_info.ResidueNumber == res:
            at_list.append(at)
        else:
            ret = Molecule()
            ret.atoms = at_list
            ret.bonds = list(set(chain.from_iterable(at.bonds for at in at_list)))
            return ret.copy()


def get_perpendicular_vec(vec: np.ndarray) -> np.ndarray:
    """Construct a unit-vector orthogonal to **vec**.

    Parameters
    ----------
    vec : :math:`n` |np.ndarray|_
        A vector represented by a 1D array-like sequence of length :math:`n`.
        The supplied vector does *not* have to be normalized.

    Returns
    -------
    :math:`n` |np.ndarray|_
        A 1D array, of length :math:`n`, representing a unit-vector orthogonal to **vec**.

    """
    one = np.ones_like(vec)
    _vec = np.asarray(vec)

    v1 = one / np.linalg.norm(one)
    v2 = _vec / np.linalg.norm(_vec)
    return v1 - v1@v2 * v2
